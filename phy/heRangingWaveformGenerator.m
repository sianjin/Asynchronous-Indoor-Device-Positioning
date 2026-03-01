function txWaveform = heRangingWaveformGenerator(cfgFormat,varargin)
%heRangingWaveformGenerator WLAN waveform generation
%   TXWAVEFORM = heRangingWaveformGenerator(CFGFORMAT) generates an HE
%   Ranging NDP waveform as in IEEE 802.11az-2022. The generated waveform
%   contains a single packet with no idle time. The function windows the
%   packet for spectral controls with a windowing transition time of 1e-7
%   seconds.
%
%   TXWAVEFORM is a complex Ns-by-Nt matrix containing the generated
%   waveform, where Ns is the number of time domain samples, and Nt is the
%   number of transmit antennas.
%
%   CFGFORMAT contains transmission parameters for the generated waveform
%   in an <a href="matlab:help('heRangingConfig')">heRangingConfig</a> object.
%
%   TXWAVEFORM = heRangingWaveformGenerator(CFGFORMAT,Name,Value) specifies
%   additional name-value pair arguments. When you do not specify a
%   name-value pair argument the function uses its default value.
%
%   'NumPackets'               Number of packets to generate, specified as
%                              a positive integer. The default value is 1.
%
%   'IdleTime'                 Length of an idle period, in seconds, after
%                              each generated packet, specified as either 0
%                              or a scalar greater than or equal to 2e-6.
%                              The default value is 0.
%
%   'OversamplingFactor'       The oversampling factor used by the
%                              function to generate the IEEE 802.11az
%                              waveform. The oversampling factor must be
%                              >=1. The resultant FFT length must be even,
%                              and the cyclic prefix length in samples must
%                              be integer-valued for all symbols within a
%                              packet. The default value is 1.
%
%   'WindowTransitionTime'     The windowing transition length in seconds,
%                              applied to the waveform. Windowing is only
%                              applied when the type of HE-LTF sequence is
%                              non-secure. The maximum window transition
%                              time is 1.6e-6 seconds. The default value is
%                              1e-7 seconds.
%
%   Examples:
%
%   Example 1:
%       %  Generate a time-domain signal for an HE Ranging NDP
%       %  transmission with 10 packets and a 20 microsecond idle period
%       %  between packets. Each packet has two repetitions of
%       %  space-time streams.
%
%       numPkts = 10;                % 10 packets in the waveform
%       cfg = heRangingConfig();     % Create format configuration
%       cfg.NumTransmitAntennas = 2; % Number of transmit antennas
%
%       % Set space-time streams and number of HE-LTF repetitions
%       cfg.User{1}.NumSpaceTimeStreams = 2;
%       cfg.User{1}.NumHELTFRepetitions = 2;
%
%       txWaveform = heRangingWaveformGenerator(cfg, ...
%           'NumPackets', numPkts, 'IdleTime', 20e-6, ...
%           'WindowTransitionTime', 1e-7);
%
%   Example 2:
%       %  Generate a time domain signal for a secure HE Ranging NDP
%       %  transmission with 10 packets and 20 microsecond idle period
%       %  between packets. Each packet has two repetitions of
%       %  space-time streams.
%
%       numPkts = 10;                    % 10 packets in the waveform
%       cfg = heRangingConfig();         % Create format configuration
%       cfg.NumTransmitAntennas = 2;     % Number of transmit antennas
%       cfg.SecureHELTF = true;          % Use secure HE-LTF sequence
%
%       % Set space-time streams and number of HE-LTF repetitions
%       cfg.User{1}.NumSpaceTimeStreams = 2;
%       cfg.User{1}.NumHELTFRepetitions = 2;
%       numOctets = numSecureHELTFOctets(cfg);
%       % The secure HE-LTF octets are represented as a int vector
%       cfg.User{1}.SecureHELTFOctets = randi([0 255], 1, numOctets(1));
%
%       txWaveform = heRangingWaveformGenerator(cfg, ...
%           'NumPackets', numPkts, 'IdleTime', 20e-6);
%
%   Example 3:
%       %  Generate a time-domain signal for a secure HE Ranging NDP
%       %  transmission for two users.
%
%       cfg = heRangingConfig(2);        % Create format configuration
%
%       txWaveform = heRangingWaveformGenerator(cfg);
%
%   Example 4:
%       %  Generate a time-domain signal for a secure HE Ranging NDP
%       %  transmission for two users. The number of space-time streams are
%       %  two and one for user 1 and 2, respectively. Generate the HE-LTF
%       %  symbols for both users by specifying a secure HE-LTF sequence.
%
%       cfg = heRangingConfig(2);    % Create format configuration
%       cfg.NumTransmitAntennas = 2; % Number of transmit antennas
%
%       % Set space-time streams and number of HE-LTF repetitions
%       cfg.User{1}.NumSpaceTimeStreams = 2;
%       cfg.User{1}.NumHELTFRepetitions = 2;
%
%       cfg.User{2}.NumSpaceTimeStreams = 1;
%       cfg.User{2}.NumHELTFRepetitions = 2;
%
%       % Get the length of the secure octets
%       numOctets = numSecureHELTFOctets(cfg);
%
%       % Set secure HE-LTF octets for user-1, which is represented as a int vector
%       cfg.User{1}.SecureHELTFOctets = randi([0 255], 1, numOctets(1));
%
%       % Set secure HE-LTF octets for user-2, which is represented as hex decimals
%       cfg.User{2}.SecureHELTFOctets = dec2hex(randi([0 15], 1, 2*numOctets(2)));
%
%       txWaveform = heRangingWaveformGenerator(cfg);
%
%   See also heRangingFieldIndices

%   Copyright 2020-2023 The MathWorks, Inc.

%#codegen

% Check number of input arguments
coder.internal.errorIf(mod(nargin,2)==0,'wlan:wlanWaveformGenerator:InvalidNumInputs');

% Validate the format configuration object is a valid type
validateattributes(cfgFormat,{'heRangingConfig'},{'scalar'},mfilename,'format configuration object');
validateConfig(cfgFormat);

% Define default values for WindowTransitionTime
winTransitTime = 1e-7;
if cfgFormat.SecureHELTF
     winTransitTime = 0; % No windowing by default for secure HE-LTF
end

% Default values
defaultParams = struct('NumPackets', 1, ...
                    'IdleTime', 0, ...
                    'WindowTransitionTime', winTransitTime, ...
                    'OversamplingFactor', 1);

if nargin==1
    useParams = defaultParams;
else
    % Extract each P-V pair
    if isempty(coder.target) % Simulation path
        p = inputParser;
        addParameter(p,'NumPackets',defaultParams.NumPackets);
        addParameter(p,'IdleTime',defaultParams.IdleTime);
        addParameter(p,'WindowTransitionTime',defaultParams.WindowTransitionTime);
        addParameter(p,'OversamplingFactor',defaultParams.OversamplingFactor);
        parse(p,varargin{:});
        useParams = p.Results;
    else % Codegen path
        pvPairs = struct('NumPackets', uint32(0), ...
                         'IdleTime', uint32(0), ...
                         'WindowTransitionTime', uint32(0), ...
                         'OversamplingFactor', uint32(1));

        % Select parsing options
        popts = struct('PartialMatching',true);

        % Parse inputs
        pStruct = coder.internal.parseParameterInputs(pvPairs,popts,varargin{:});

        % Get values for the P-V pair or set defaults for the optional arguments
        useParams = struct;
        useParams.NumPackets = coder.internal.getParameterValue(pStruct.NumPackets,defaultParams.NumPackets,varargin{:});
        useParams.IdleTime = coder.internal.getParameterValue(pStruct.IdleTime,defaultParams.IdleTime,varargin{:});
        useParams.WindowTransitionTime = coder.internal.getParameterValue(pStruct.WindowTransitionTime,defaultParams.WindowTransitionTime,varargin{:});
        useParams.OversamplingFactor = coder.internal.getParameterValue(pStruct.OversamplingFactor,defaultParams.OversamplingFactor,varargin{:});
    end

    if cfgFormat.SecureHELTF && useParams.WindowTransitionTime > 0 % Do not window HE ranging NDP with secure HE-LTF sequence
        error('Windowing is not supported for HE-LTF symbols with secure HE-LTF sequence');
    end

    % Validate each P-V pair
    % Validate numPackets
    validateattributes(useParams.NumPackets,{'numeric'},{'scalar','integer','>=',0},mfilename,'''NumPackets'' value');
    % Validate idleTime
    validateattributes(useParams.IdleTime,{'numeric'},{'scalar','real','>=',0},mfilename,'''IdleTime'' value');
    minIdleTime = 2e-6;
    coder.internal.errorIf((useParams.IdleTime > 0) && (useParams.IdleTime < minIdleTime),'wlan:wlanWaveformGenerator:InvalidIdleTimeValue',sprintf('%1.0d',minIdleTime));
    % Validate WindowTransitionTime
    maxWinTransitTime = 1.6e-6; % Seconds
    validateattributes(useParams.WindowTransitionTime,{'numeric'},{'real','scalar','>=',0,'<=',maxWinTransitTime},mfilename,'''WindowTransitionTime'' value');
    % Validate OversamplingFactor
    validateattributes(useParams.OversamplingFactor,{'numeric'},{'real','finite','scalar','>=',1},mfilename,'OversamplingFactor');
end

osf = double(useParams.OversamplingFactor); % Force from numeric to double

% Get the sampling rate of the waveform
[psps,trc] = hePacketSamplesPerSymbol(cfgFormat,osf);
numPktSamples = psps.NumPacketSamples;
numTxAnt = cfgFormat.NumTransmitAntennas;
cbw = wlan.internal.cbwStr2Num(cfgFormat.ChannelBandwidth);
sr = cbw*1e6*osf;
sf = cbw*1e-3*osf; % Scaling factor to convert bandwidth and time in ns to samples
cpLen = trc.TGIData*sf;
heltfSymLen = trc.THELTFSYM*sf;
Npe = wlan.internal.heNumPacketExtensionSamples(trc.TPE,cbw)*osf;

LSTF = wlan.internal.heLSTF(cfgFormat,osf);
LLTF = wlan.internal.heLLTF(cfgFormat,osf);
LSIG = wlan.internal.heLSIG(cfgFormat,osf);
RLSIG = LSIG; % RL-SIG is identical to L-SIG
SIGA = wlan.internal.heSIGA(cfgFormat,osf);
% The number of transmit antenna for HE-STF should match the number of
% transmit antenna in the first HE-LTF user block. IEEE 802.11az-2022
% Section 27.3.18a.1. The wlan.internal.heSTF takes the max of the number
% of space-time streams between all users. A dummy, single user
% heRangingConfig object with same user properties as cfgFormat in used to
% generate HE-STF samples.
cfgHESTF = cfgFormat;
cfgHESTF.User = {cfgFormat.User{1}};
STF = wlan.internal.heSTF(cfgHESTF,osf);
LTF = heRangingLTF(cfgFormat,osf);
preamble = [LSTF; LLTF; LSIG; RLSIG; SIGA; STF; LTF];

% Define a matrix of total simulation length
numIdleSamples = round(sr*useParams.IdleTime);
pktWithIdleLength = numPktSamples+numIdleSamples;
txWaveform = complex(zeros(useParams.NumPackets*pktWithIdleLength,numTxAnt));

% Set the PE field
lastDataSymBlk = preamble(end-heltfSymLen+cpLen+1:end,:);
packetExt = getPacketExtensionData(lastDataSymBlk,Npe);
if cfgFormat.SecureHELTF
    peField = [zeros(cpLen,size(packetExt,2)); packetExt(cpLen+1:end,:)]; % Zero power GI
else
    peField = packetExt;
end
    
for i = 1:useParams.NumPackets
    % Construct packet from preamble and data
    packet = [preamble; peField];
    txWaveform((i-1)*pktWithIdleLength+(1:numPktSamples),:) = packet;
end

if useParams.NumPackets>0 && useParams.WindowTransitionTime>0
    % Window waveform (no need to validate TT against GI dynamically)
    wLength = 2*ceil(useParams.WindowTransitionTime*sr/2);
    txWaveform = wlan.internal.windowWaveform(txWaveform,psps.NumSamplesPerSymbol,psps.CPPerSymbol,psps.ExtensionPerSymbol,wLength,useParams.NumPackets,numIdleSamples);
end
end

function packetExt = getPacketExtensionData(lastDataSymBlk,Npe)
    % Cyclic extension of last symbol for packet extension
    if size(lastDataSymBlk,1)>=Npe
        packetExt = lastDataSymBlk(1:Npe,:);
    else
        buffCntt = ceil(Npe/size(lastDataSymBlk,1));
        dataBuffer = repmat(lastDataSymBlk,buffCntt,1);
        packetExt = dataBuffer(1:Npe,:);
    end
end

function [Y,trc] = hePacketSamplesPerSymbol(cfg,varargin)
%hePacketSamplesPerSymbol Samples per symbol information

osf = 1;
if nargin>1
    osf = varargin{1};
end

cbw = wlan.internal.cbwStr2Num(cfg.ChannelBandwidth);
sf = cbw*1e-3*osf; % Scaling factor to convert bandwidth and time in ns to samples
commonCodingParams = wlan.internal.heCodingParameters(cfg);
numDataSym = commonCodingParams.NSYM;
npp = wlan.internal.heNominalPacketPadding(cfg);
trc = wlan.internal.heTimingRelatedConstants(cfg.GuardInterval,cfg.HELTFType,commonCodingParams.PreFECPaddingFactor,npp,commonCodingParams.NSYM);

% For each format calculate the number of samples per symbol and CP in the
% preamble fields which are format dependent
numHELTFSym = numHELTFSymbols(cfg);
numSIGASym = 2;
samplesPerSymbolPreamble = [trc.THESIGA/numSIGASym*ones(1,numSIGASym) trc.THESTFNT];
cpLengthPreamble = [trc.TGILegacyPreamble*ones(1,2) 0];

% Midamble periodicity. IEEE 802.11ax-2021, Section 27.3.12.16
Mma = cfg.MidamblePeriodicity;
Nma = wlan.internal.numMidamblePeriods(cfg,numDataSym); % Midamble period
if Nma>0
    nonCPData = trc.TSYM*ones(1,numDataSym);
    nonCPHELTF = trc.THELTFSYM*ones(1,numHELTFSym);
    % Reshape data symbols till last midamble in to data symbol blocks
    data = reshape(nonCPData(1:Mma*Nma),Mma,Nma);
    % Repeat HELTF symbols for each data symbol block
    heLTF = repmat(nonCPHELTF,Nma,1).';
    % Append midamble after each data symbol block
    dataWithHELTF = [data; heLTF];
    % Reshape and append leftover data samples after the last midamble
    samplesPerSymbolData = [dataWithHELTF(:).' nonCPData(Mma*Nma+1:end)];

    % Cyclic prefix field
    cpLengthData = trc.TGIData*ones(1,numDataSym+numHELTFSym*Nma);
else
    samplesPerSymbolData = trc.TSYM*ones(1,numDataSym);
    cpLengthData = trc.TGIData*ones(1,numDataSym);
end

if numDataSym>0
    % Create a vector of the samples per symbol
    samplesPerSymbol = [trc.TLSTF trc.TLLTF trc.TLSIG trc.TRLSIG ...
        samplesPerSymbolPreamble ...
        trc.THELTFSYM*ones(1,numHELTFSym) ...
        samplesPerSymbolData(1:end-1) samplesPerSymbolData(end)+trc.TPE]*sf;
else
    % Create a vector of the samples per symbol
    samplesPerSymbol = [trc.TLSTF trc.TLLTF trc.TLSIG trc.TRLSIG ...
        samplesPerSymbolPreamble ...
        trc.THELTFSYM*ones(1,numHELTFSym-1) (trc.THELTFSYM+trc.TPE)]*sf;
end

% Create a vector of the CP per symbol
cpPerSymbol = [0 trc.TGILLTF trc.TGILegacyPreamble trc.TGILegacyPreamble ...
    cpLengthPreamble ...
    trc.TGIHELTF*ones(1,numHELTFSym) ...
    cpLengthData]*sf;

numPPDUSamples = sum(samplesPerSymbol);
Y = struct('NumSamplesPerSymbol', samplesPerSymbol, ...
           'CPPerSymbol', cpPerSymbol, ...
           'ExtensionPerSymbol', [zeros(1,length(samplesPerSymbol)-1) trc.TPE*sf], ...
           'NumPacketSamples', numPPDUSamples);

end
