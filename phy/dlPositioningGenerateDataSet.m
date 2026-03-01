function [cir, labels] = dlPositioningGenerateDataSet(rays, STAs, APs, cfg, snrs)
%dlPositioningGenerateDataSetComplex Generate a Dataset of Complex CIR Fingerprints 
%   (Stacked Real/Imaginary parts)

ofdmSymbolOffset = 0.75;
numChan = numel(rays);
txWaveform = single(heRangingWaveformGenerator(cfg)); 
ofdmInfo = wlanHEOFDMInfo('HE-LTF',cfg.ChannelBandwidth,cfg.GuardInterval);

% --- MODIFICATION 1: Double the spatial dimension for [Real; Imag] ---
numSpatialStreams = cfg.User{1}.NumSpaceTimeStreams * prod(STAs(1).Antenna.Size);
cir = zeros([ofdmSymbolOffset*ofdmInfo.CPLength 2*numSpatialStreams length(snrs) numChan], 'single');

labels.position = zeros([3 numChan]);
for i = 1:numChan
    txn = mod(i-1,height(rays))+1;
    rxn = ceil(i/height(rays));
    if isempty(rays{i})
        cir(:,:,:,i) = 0;    
    else
        % Generates the complex channel estimate
        cir(:,:,:,i) = generateComplexCIR(rays{i},APs(txn),STAs(rxn),cfg,txWaveform,ofdmInfo,snrs,ofdmSymbolOffset);     
    end       
    labels.position(:,i) = [STAs(rxn).AntennaPosition];
    labels.class(i) = categorical(cellstr(STAs(rxn).Name));  

    if mod(i,floor(numChan/10))==0
        qt = ceil(i/(numChan/10));
        disp(['Generating Complex Dataset: ', num2str(10*qt), '% complete.'])
    end
end

% Reshape/Permute logic remains identical to original (dimensions auto-adjust)
cir = reshape(cir,[ofdmSymbolOffset*ofdmInfo.CPLength 2*numSpatialStreams length(snrs) numel(APs) numel(STAs)]);
cir = permute(cir,[1 2 4 3 5]);
cir = reshape(cir,[size(cir,1) size(cir,2) size(cir,3) size(cir,4)*size(cir,5)]);

labels.position = labels.position(:, 1:height(rays):end);
labels.class = labels.class(:, 1:height(rays):end); 
labels.position = repelem(labels.position, 1, length(snrs));
labels.class = repelem(labels.class, 1, length(snrs));

end

function cir = generateComplexCIR(rays, AP, STA, cfg, tx, ofdmInfo, snr, ofdmSymbolOffset)
% --- 1. Fast Parameter Setup (Double Precision) ---
fs = double(ofdmInfo.SampleRate);
fc = double(AP.TransmitterFrequency);

% Hardware Offsets (Randomized for training)
sco = unifrnd(-100, 100);
sro = -sco / (1 + sco/1e6);
cfo = (sco * 1e-6) * fc + (randn * 2);

% Velocity (Walking speed)
v_mag = unifrnd(0, 1.5);
v_vec = (v_mag * randn(3,1));
v_vec = v_vec / (norm(v_vec) + 1e-9) * v_mag;

% --- 2. Ray Tracing (The Physics Engine) ---
rtChan = comm.RayTracingChannel(rays, AP, STA);
rtChan.SampleRate = fs;
rtChan.ReceiverVirtualVelocity = v_vec;
rtChan.NormalizeChannelOutputs = false;
rxChanClean = double(rtChan(double(tx))); % Output is [Samples x Nr]

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
pn_std = unifrnd(0.005, 0.02); 
rxImpaired = rxImpaired .* exp(1j * cumsum(pn_std * randn(size(rxImpaired,1), 1)));

% --- 4. Feature Extraction & SNR Loop ---
numSpatialStreams = cfg.User{1}.NumSpaceTimeStreams * prod(STA.Antenna.Size);
numTaps = floor(ofdmSymbolOffset * ofdmInfo.CPLength);

% Dimension 2 is 2*Nr because we stack Real and Imaginary
cir = zeros([numTaps, 2*numSpatialStreams, length(snr)], 'single');
snrAdj = snr - 10*log10(ofdmInfo.FFTLength / ofdmInfo.NumTones);

for i = 1:length(snr)
    % Add Noise
    rxNoisy = awgn(rxImpaired, snrAdj(i));

    % Sync & Channel Estimate
    chanEst = heRangingSynchronize(rxNoisy, cfg);
    if isempty(chanEst), continue; end

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
    % This is where the CNN tries to find patterns in the phase
    cir(:, 1:numSpatialStreams, i) = real(temp);
    cir(:, numSpatialStreams+1:end, i) = imag(temp);
end
end