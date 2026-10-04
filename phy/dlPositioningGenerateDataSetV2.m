function [cir, labels] = dlPositioningGenerateDataSetV2(rays, STAs, APs, cfg, snrs, imp)
%dlPositioningGenerateDataSetV2 Generate a Dataset of Complex CIR Fingerprints
%   (Stacked Real/Imaginary parts) with configurable hardware impairments
%
%   [CIR,LABELS] = dlPositioningGenerateDataSetV2(RAYS,STAS,APS,CFG,SNRS,IMP)
%   differs from dlPositioningGenerateDataSet in three ways:
%     1. The hardware impairments are set by the structure IMP and are drawn
%        independently for every sample (every AP, STA and SNR), instead of
%        once per AP-STA link.
%     2. LABELS additionally records, per sample, the SNR, and per AP and
%        sample, the line-of-sight flag, the packet detection flag and the
%        impairment realization.
%     3. No training/validation split is applied to the output order: sample
%        (rxn-1)*numel(SNRS)+s is STA rxn at SNRS(s).
%
%   IMP is a structure with fields:
%     ClockOffsetPPM  - Maximum relative clock offset in ppm. The offset is
%                       uniform in [-ClockOffsetPPM, ClockOffsetPPM] and sets
%                       both the sampling clock offset and the CFO.
%     CFOJitterStd    - Standard deviation in Hz of the CFO component that is
%                       not explained by the clock offset.
%     PhaseNoiseStd   - [min max] of the per-sample standard deviation in
%                       radians of the random-walk phase noise increment.
%     MaxSpeed        - Maximum STA speed in m/s (uniform in [0, MaxSpeed]).
%   Set a field to 0 to disable that impairment.
%
%   CIR is a Ntap-by-2*Nsts*Nr-by-numAPs-by-numSTAs*numSNRs array.

ofdmSymbolOffset = 0.75;
[numAPs, numSTAs] = size(rays);
numSNRs = length(snrs);
txWaveform = double(heRangingWaveformGenerator(cfg));
ofdmInfo = wlanHEOFDMInfo('HE-LTF',cfg.ChannelBandwidth,cfg.GuardInterval);

numSpatialStreams = cfg.User{1}.NumSpaceTimeStreams * prod(STAs(1).Antenna.Size);
numTaps = floor(ofdmSymbolOffset*ofdmInfo.CPLength);
cir = zeros([numTaps 2*numSpatialStreams numAPs numSNRs numSTAs], 'single');

los = false([numAPs numSNRs numSTAs]);
detected = false([numAPs numSNRs numSTAs]);
clockOffsetPPM = zeros([numAPs numSNRs numSTAs]);
phaseNoiseStd = zeros([numAPs numSNRs numSTAs]);
position = zeros([3 numSTAs]);
class = categorical(strings(1,numSTAs));

for rxn = 1:numSTAs
    position(:,rxn) = STAs(rxn).AntennaPosition;
    class(rxn) = categorical(cellstr(STAs(rxn).Name));
    for txn = 1:numAPs
        r = rays{txn,rxn};
        if isempty(r)
            continue
        end
        los(txn,:,rxn) = any([r.LineOfSight]);
        for s = 1:numSNRs
            % Independent impairment realization for every sample
            [c, d, p] = generateComplexCIR(r,APs(txn),STAs(rxn),cfg,txWaveform,ofdmInfo,snrs(s),numTaps,imp);
            cir(:,:,txn,s,rxn) = c;
            detected(txn,s,rxn) = d;
            clockOffsetPPM(txn,s,rxn) = p.ClockOffsetPPM;
            phaseNoiseStd(txn,s,rxn) = p.PhaseNoiseStd;
        end
    end

    if mod(rxn,ceil(numSTAs/10))==0
        disp(['Generating Complex Dataset: ', num2str(round(100*rxn/numSTAs)), '% complete.'])
    end
end

% Merge the SNR and STA dimensions into one sample dimension
numSamples = numSNRs*numSTAs;
cir = reshape(cir,[numTaps 2*numSpatialStreams numAPs numSamples]);

labels.position = repelem(position, 1, numSNRs);
labels.class = repelem(class, 1, numSNRs);
labels.snr = repmat(snrs(:).', 1, numSTAs);
labels.los = reshape(los,[numAPs numSamples]);
labels.detected = reshape(detected,[numAPs numSamples]);
labels.clockOffsetPPM = reshape(clockOffsetPPM,[numAPs numSamples]);
labels.phaseNoiseStd = reshape(phaseNoiseStd,[numAPs numSamples]);

end

function [cir, detected, p] = generateComplexCIR(rays, AP, STA, cfg, tx, ofdmInfo, snr, numTaps, imp)
% --- 1. Fast Parameter Setup (Double Precision) ---
fs = double(ofdmInfo.SampleRate);
fc = double(AP.TransmitterFrequency);

% Hardware Offsets (Randomized per sample)
sco = unifrnd(-imp.ClockOffsetPPM, imp.ClockOffsetPPM);
sro = -sco / (1 + sco/1e6);
cfo = (sco * 1e-6) * fc + (randn * imp.CFOJitterStd);

% Velocity (Walking speed)
v_mag = unifrnd(0, imp.MaxSpeed);
v_vec = randn(3,1);
v_vec = v_vec / (norm(v_vec) + 1e-9) * v_mag;

% --- 2. Ray Tracing (The Physics Engine) ---
rtChan = comm.RayTracingChannel(rays, AP, STA);
rtChan.SampleRate = fs;
rtChan.ReceiverVirtualVelocity = v_vec;
rtChan.NormalizeChannelOutputs = false;
rxChanClean = double(rtChan(tx)); % Output is [Samples x Nr]

% --- 3. Fast Impairments (Vectorized & Stochastic) ---

% A. Faster Resampling for SCO (Linear Interpolation)
t_orig = (0:size(rxChanClean,1)-1)';
t_new = t_orig * (1 + sro/1e6);
rxImpaired = interp1(t_orig, rxChanClean, t_new, 'linear', 0);

% B. Vectorized CFO
t = (0:size(rxImpaired,1)-1)' / fs;
rxImpaired = rxImpaired .* exp(1j * 2 * pi * cfo * t);

% C. Fast Random Walk Phase Noise
% Models intrinsic LO jitter that survive synchronization
pn_std = unifrnd(imp.PhaseNoiseStd(1), imp.PhaseNoiseStd(end));
rxImpaired = rxImpaired .* exp(1j * cumsum(pn_std * randn(size(rxImpaired,1), 1)));

p.ClockOffsetPPM = sco;
p.PhaseNoiseStd = pn_std;

% --- 4. Feature Extraction ---
numSpatialStreams = cfg.User{1}.NumSpaceTimeStreams * prod(STA.Antenna.Size);

% Dimension 2 is 2*Nr*Nsts because we stack Real and Imaginary
cir = zeros([numTaps, 2*numSpatialStreams], 'single');
snrAdj = snr - 10*log10(ofdmInfo.FFTLength / ofdmInfo.NumTones);

% Add Noise
rxNoisy = awgn(rxImpaired, snrAdj);

% Sync & Channel Estimate
chanEst = heRangingSynchronize(rxNoisy, cfg);
detected = ~isempty(chanEst);
if ~detected, return; end

% Get raw complex CIR
cirRaw = helperChannelImpulseResponse(single(chanEst), ...
    ofdmInfo.FFTLength, ofdmInfo.CPLength, ofdmInfo.ActiveFFTIndices);

% Trim and Flatten
L = min(size(cirRaw, 1), numTaps);
temp = zeros(numTaps, numSpatialStreams, 'single');
% Combine all spatial dimensions (Nr * Nsts)
cirRawReshaped = reshape(cirRaw, size(cirRaw,1), []);
temp(1:L, :) = cirRawReshaped(1:L, :);

% Stack Real and Imaginary parts
cir(:, 1:numSpatialStreams) = real(temp);
cir(:, numSpatialStreams+1:end) = imag(temp);
end
