function [HELTF,kHELTFSeq,kNz,phaseRotation] = heSecureLTFSequence(cfg,varargin)
%heSecureLTFSequence Secure HE-LTF symbol parameters
%   [HELTF,kHELTFSeq,kNz,phaseRotation] = heSecureLTFSequence(cfg,varargin) 
%   returns these secure HE-LTF symbol parameters.
% 
%   - HELTF: A matrix of size N-by-NHELTF containing the secure HE-LTF
%     sequence mapped to the corresponding subcarrier indices. N is the
%     number of subcarriers for the corresponding bandwidth and NHELTF is
%     the number of HE-LTF symbols.
%
%   - kHELTFSeq: A column vector of corresponding subcarrier indices
%
%   - kNz: A column vector of non-zero subcarrier indices
%
%   - phaseRotation: A complex matrix of size NSTS-by-NHELTF containing the
%     phase shift for each space-time stream and each HE-LTF. NSTS is the
%     number of space time streams and NHELTF is the number of HE-LTF
%     symbols.
%
%   cfg is a format configuration object of type <a href="matlab:help('heRangingConfig')">heRangingConfig</a>.
%
%   [...] = heSecureLTFSequence(...,UserNumber) returns the 
%   HELTF,kHELTFSeq,kNz,phaseRotation for a given user whose index is UserNumber.

%   Copyright 2020-2023 The MathWorks, Inc.

%#codegen

narginchk(1,2);
validateattributes(cfg,{'heRangingConfig'},{'scalar'},mfilename,'Configuration object');

% Find the non-zero entries from the regular 2x HE-LTF as IEEE 802.11az-2022,
% Section 27.3.18b.2.
% Get the non-secure HE-LTF and find the non-zero entries.
cbw = wlan.internal.cbwStr2Num(cfg.ChannelBandwidth);
[nonSecureHELTF,kHELTFSeq] = wlan.internal.heLTFSequence(cbw,cfg.HELTFType);
idxNz = find(nonSecureHELTF);
kNz = kHELTFSeq(idxNz);
numNz = size(kNz,1);
heLTFInfo = heRangingLTFInfo(cfg);

% Get number of users
numUsers = heLTFInfo.NUsers;

if nargin>1
    startUserIdx = varargin{1};
    if ~isnumeric(startUserIdx)
        error('Second input argument must be numeric scalar.');
    end

    if startUserIdx>numUsers
        error('The UserNumber must not be greater than the total number of users in HE-LTF field.');
    end
    endUserIdx = startUserIdx; % Return the secure sequence for only one user, which is the specified using the input.
else
    startUserIdx = 1;
    endUserIdx = numUsers;
end

numOutputHELTF = sum(heLTFInfo.NHELTFWithRepetition(startUserIdx:endUserIdx));
HELTF = complex(zeros(length(kHELTFSeq),numOutputHELTF));
phaseRotation = coder.nullcopy(complex(zeros(cfg.NumSpaceTimeStreams,numOutputHELTF)));

% Index of current symbol in the whole sequence.
symIdx = 0;
for u=startUserIdx:endUserIdx % Process HE-LTF user blocks
    numReps = heLTFInfo.NRepetitions(u);
    numSTS = heLTFInfo.NSTS(u);
    numSymsPerRep = heLTFInfo.NHELTFWithoutRepetition(u);

    % 7 (for phase rotation) + (number of valid subcarriers * number of
    % symbols)* number of repetition
    numRequriredOctets = 7+(numNz*numSymsPerRep)*numReps;
    if isa(cfg.User{u}.SecureHELTFOctets,'numeric')
        octets = uint8(cfg.User{u}.SecureHELTFOctets);
    else
        % Convert string to a uint8 vector
        hexArray = reshape(cfg.User{u}.SecureHELTFOctets,2,[])';
        octets = uint8(hex2dec(hexArray));
    end

    % Crop or repeat the octets as necessary.
    userOctets = octets(mod((0:numRequriredOctets-1)',length(octets))+1);
    userOctets = reshape(userOctets,1,[]);

    for r=1:numReps % HE-LTF Repetition block
        % Calculate the phase rotation.
        % Starting from stream 2 since the first rotation is always 0 (1+0i
        % in complex format) as 802.11az-2022 Section 27.3.18b.3
        blockPhaseRotation = complex(ones(heLTFInfo.MaxNSTS,1));
        if numSTS>1
            s=2:numSTS;
            blockPhaseRotation(s)=exp(1i*(pseudorandomPhaseRotation(userOctets(s-1)) ... % pseudorandomPhaseRotation is same for all repetitions within a user block, Equation 27-126d
                +deterministicPhaseRotation(r,s)));
        end

        for n=1:numSymsPerRep
            % Create the LTF symbols, using octets starting from octet 8
            % since the first 7 octets are used for phase rotation.
            symOctet = userOctets(7+(r-1)*numNz*numSymsPerRep+(n-1)*numNz+(1:numNz));
            bitstream = int2bit(symOctet,6,false);
            bitstream = int8(bitstream(:));

            % The bit sequence of wlanConstellationMap is B0 B1 B2 B3 B4 B5.
            % The secound parameter is set to 6 for 64-QAM.
            syms = wlanConstellationMap(bitstream,6);

            if cbw == 160
                % A segment parser is required to process it as upper and lower
                % segments when using 160MHz.
                % Table 27-47d shows the mapping and it is implmented here.
                symsCopy = syms;
                syms(1:length(syms)/2) = symsCopy(1:2:end); % The lower 80 MHz segment uses the symbols with odd number indicies.
                syms(length(syms)/2+1:end) = symsCopy(2:2:end); % The upper 80 MHz segment uses the symbols with even number indicies.
            end

            % Increas the index by one then set the symbol and phase
            % rotation.
            symIdx = symIdx+1;
            HELTF(idxNz,symIdx) = syms;
            phaseRotation(:,symIdx) = blockPhaseRotation;
        end
    end
end

function phase = deterministicPhaseRotation(r,s)
%phase Return the deterministic phase roation
    persistent betaTbl % keep the table in memory between function calls
    if isempty(betaTbl)
        betaTbl = pi/4*...
        [0, 0, 0, 0, 0, 0, 0, 0;
         0, 2, 3, 5, 4, 6, 7, 1;
         0, 3, 5, 1, 4, 7, 2, 6;
         0, 7, 6, 4, 1, 5, 3, 2;
         0, 1, 4, 6, 7, 5, 3, 2;
         0, 6, 1, 2, 7, 4, 5, 3;
         0, 4, 7, 3, 2, 1, 5, 6;
         0, 5, 2, 7, 1, 4, 6, 3;]; % Table 27-47f
    end
    phase = betaTbl(r,s);
end

function phase = pseudorandomPhaseRotation(octet)
%phase Return the pseudorandom phase roation
    bits = int2bit(octet,8,false);
    k = double((4*bits(6,:))+(2*bits(7,:))+bits(8,:)); % Equation 27-126e
    phase = k*pi/4; % Equation 27-126f
end

end