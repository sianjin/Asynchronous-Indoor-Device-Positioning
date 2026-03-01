function y = heRangingLTF(cfg,osf)
%heRangingLTF HE Ranging NDP Long Training Field (HE-LTF)
%   Y = heRangingLTF(CFG) generates an HE Ranging Long Training Field
%   (HE-LTF) time-domain signal for the HE Ranging NDP format as defined in
%   IEEE 802.11az-2022 standard.
%
%   Y is the time-domain HE-LTF signal. It is a complex matrix of size
%   Ns-by-Nt where Ns represents the number of time-domain samples and Nt
%   represents the number of transmit antennas.
%
%   CFG is the format configuration object of type <a href="matlab:help('heRangingConfig')">heRangingConfig</a>
%
%   Y = heLTF(CFG,OSF) generates the HE-LTF for the given oversampling
%   factor OSF. When not specified 1 is assumed.
%
%   % Example: Generate an HE Ranging LTF field for a 80MHz.
%
%   cfg = heRangingConfig('ChannelBandwidth','CBW80');
%   y = heRangingLTF(cfg);
%   plot(abs(y));

%   Copyright 2020-2025 The MathWorks, Inc.

%#codegen

validateattributes(cfg,{'heRangingConfig'},{'scalar'},mfilename,'Configuration object');
[~,heLTFInfo] = validateConfig(cfg);
numTx = cfg.NumTransmitAntennas;
cbw = wlan.internal.cbwStr2Num(cfg.ChannelBandwidth);
numUsers = heLTFInfo.NUsers; % Number of user in HE-LTF field
allocationInfo = ruInfo(cfg); % RU allocation info
Nltf = heLTFInfo.NHELTF; % Number of HE-LTF symbols
maxNumSpaceTimeStream = heLTFInfo.MaxNSTS; % Max number of space-time streams between all users

if nargin<2
    osf = 1;
end

phaseRotation = zeros(1,1); % For codegen
kNz = zeros(1,0); % For codegen
secureMode = cfg.SecureHELTF;
if secureMode
    % Get HE-LTF secure sequence. IEEE 802.11az-2022, Section 27.3.18b
    [HELTF,kHELTFSeq,kNz,phaseRotation] = heSecureLTFSequence(cfg); % HELTF is of size kHELTFSeq-by-Nltf
else
    % Get HE-LTF sequence. IEEE 802.11ax-2021, Section 27.3.11.10
    [HELTF,kHELTFSeq] = wlan.internal.heLTFSequence(cbw,cfg.HELTFType); % HELTF is of size kHELTFSeq-by-1
end

Nfft = 256*cbw/20;
ofdmGrid = complex(zeros(Nfft,Nltf,numTx));
kRUPuncture = wlan.internal.hePuncturedRUSubcarrierIndices(cfg);

% Extract the RU of interest from the full-bandwidth HE-LTF
kRUFull = wlan.internal.heRUSubcarrierIndices(cbw,allocationInfo.RUSizes(1),allocationInfo.RUIndices(1));
if ~isempty(kRUPuncture)
    kRU = setdiff(kRUFull,kRUPuncture); % Discard punctured subcarriers
else
    kRU = kRUFull;
end
[~,ruIdx] = intersect(kHELTFSeq,kRU);
HELTFRU = HELTF(ruIdx,:);

% Create mapping matrix for all HE-LTF symbols per user
% Generate P and R matrix for all HE-LTF symbols within an HE-LTF field
PRep = complex(zeros(maxNumSpaceTimeStream,Nltf));
RRep = complex(zeros(maxNumSpaceTimeStream,Nltf));
heltfSymOffset = [0 cumsum(heLTFInfo.NHELTFWithRepetition)];
numLTF = heLTFInfo.NHELTFWithoutRepetition;

powerBoostFactor = 1; % No additional power boost (alpha)
cardKr = numel(kRU);
cardKHELTFr = cardKr/cfg.HELTFType;
userSTSScalingFactor = zeros(1,Nltf);
scalingConstant = powerBoostFactor*sqrt(cardKr);

for u = 1:numUsers
    % Mapping matrix for all symbols and users. IEEE 802.11ax-2021,
    % Equation 27-57
    numSTS = heLTFInfo.NSTS(u);
    numRep = heLTFInfo.NRepetitions(u);
    % Calculate P and R values in a matrix
    Pheltf = wlan.internal.mappingMatrix(numLTF(u));
    % Get P matrix for the number of HE-LTF symbols
    P = Pheltf((1:numSTS).',1:numLTF(u));
    % Same P matrix is used for all the repeated HE-LTF symbols
    P = repmat(P,1,numRep);
    PRep(1:numSTS,(1:heLTFInfo.NHELTFWithRepetition(u))+heltfSymOffset(u)) = P;
    % Get R matrix for the number of space-time streams
    R = repmat(Pheltf(1,1:numLTF(u)),numSTS,1);
    % Same R matrix is used for all the repeated HE-LTF symbols
    R = repmat(R,1,numRep);
    RRep(1:numSTS,(1:heLTFInfo.NHELTFWithRepetition(u))+heltfSymOffset(u)) = R;

    % STS scaling per user
    % Calculate per RU scaling as per IEEE 802.11ax-2021, Section 27.3.10,
    % Equation 27-5. Note we scale by sqrt(cardK)/sqrt(cardKHELTFr). This
    % causes us to normalize the power in the RU. The HE-LTF symbols for
    % the user are scalled by the number of space-time streams for the user
    % of interest.
    ruScalingFactor = scalingConstant/sqrt(cfg.User{u}.NumSpaceTimeStreams*cardKHELTFr);
    userSTSScalingFactor(1,(1:heLTFInfo.NHELTFWithRepetition(u))+heltfSymOffset(u)) = repmat(ruScalingFactor,1,heLTFInfo.NHELTFWithRepetition(u));
end

ruGrid = complex(zeros(numel(kRU),Nltf,maxNumSpaceTimeStream));
% Indices of data and pilot subcarriers within the occupied RU
kPilot = wlan.internal.hePilotSubcarrierIndices(cbw,allocationInfo.RUSizes);
ruInd = wlan.internal.heOccupiedSubcarrierIndices(kRU,kPilot);

% Indices of data and pilot subcarriers within the occupied RU
for k = 1:Nltf
    if secureMode 
        % Data is multiplied by P matrix so whole RU multiplied by P
        % matrix. Then the per symbol phase rotation is applied.
        ruGrid(:,k,:) = squeeze(HELTFRU(:,k,:)).*PRep(:,k).'.*phaseRotation(:,k).';
    else
        % Single stream pilots
        ruGrid(ruInd.Data,k,:) = HELTFRU(ruInd.Data).*PRep(:,k).';
        ruGrid(ruInd.Pilot,k,:) = HELTFRU(ruInd.Pilot).*RRep(:,k).';
    end
end

if ~secureMode
    % Cyclic shift is applied per space time stream
    ruGrid = wlan.internal.heCyclicShift(ruGrid,cbw,kRU,(1:maxNumSpaceTimeStream).');
end

% Map subcarriers to full FFT grid and scale
n_Scale = 1; % Scaling factor for an HE SU format. IEEE 802.11ax-2021, Section 27.3.11.10
ofdmGrid(kRU+Nfft/2+1,:,1:maxNumSpaceTimeStream) = ruGrid.*userSTSScalingFactor*n_Scale;

% Overall scaling factor
allScalingFactor = Nfft/sqrt(sum(powerBoostFactor.^2.*cardKr));

CPLenSamples = cfg.GuardInterval*cbw;

if secureMode && cfg.FrequencyDomainWindowing % Apply the frequency domain windowing if required
    fdw = heRangingFrequencyDomainWindow(Nfft,kNz);
    ofdmGrid = ofdmGrid.*fdw;
end

if secureMode % OFDM modulate with zero power GI
    NfftOsf = round(osf*Nfft/2);
    CPLenSamplesOsf = round(osf*CPLenSamples);
    ofdmGridOsf = complex(zeros(NfftOsf,Nltf,numTx));
    ofdmGridOsf((1:Nfft/2)+(NfftOsf-Nfft/2)/2,:,:) = ofdmGrid(1:2:end,:,:);
    postShift = ifftshift(ofdmGridOsf,1);
    postIFFT = ifft(postShift,[],1);
    postCP = [complex(zeros(CPLenSamplesOsf,Nltf,numTx)); postIFFT]; % Add zero power CP
    y = reshape(postCP,[(NfftOsf+CPLenSamplesOsf)*Nltf numTx]).*allScalingFactor/2*osf; %/2 to account for NFFT/2 actual FFT size, osf to account for oversampling
else
    y = wlan.internal.ofdmModulate(ofdmGrid(1:2:end,:,:),CPLenSamples,osf).*allScalingFactor/2; %/2 to account for NFFT/2 actual FFT size
end

end
