function hezInfo = heRangingLTFInfo(cfg)
%heRangingLTFInfo HE-LTF information of Ranging NDP packet
%   hezInfo = heRangingLTFInfo(cfg) returns a structure containing
%   HE-LTF field information for the format configuration object cfg.
%   The structure hezInfo has the following fields:
%
%   NUsers                      - Number of users.
%
%   NSTS                        - Number of space-time streams as
%                                 a vector of size 1-by-NumUsers.
%
%   MaxNSTS                     - Maximum  number of space-time streams
%                                 across all users.
%
%   NRepetitions                - Number of repetitions as a vector of size 1-by-NumUsers.
%
%   NHELTF                      - Number of HE-LTF symbols in HE-LTF.
%
%   NHELTFWithRepetition        - Number of HE-LTF symbols in HE-LTF
%                                 including repetition of HE-LTF
%                                 symbols for all users as a vector of size
%                                 1-by-NumUsers.
%
%   NHELTFWithoutRepetition     - Number of HE-LTF symbols in HE-LTF
%                                 excluding repetition of HE-LTF symbols as
%                                 a vector of size 1-by-NumUsers.
%
%   NSecureHELTFOctets          - Number of secure octets required to encode
%                                 the HE-LTF symbols for all users as a
%                                 vector of size 1-by-NumUsers.
%
%   cfg is a format configuration object of type <a href="matlab:help('heRangingConfig')">heRangingConfig</a>.

%   Copyright 2020-2023 The MathWorks, Inc.

%#codegen

validateattributes(cfg,{'heRangingConfig'},{'scalar'},mfilename,'Configuration object');
cbw = wlan.internal.cbwStr2Num(cfg.ChannelBandwidth);
switch cbw
    case 20
        numNz = 122; % Number of non-zero subcarriers as defined in Equation 27-126a of IEEE Std 802.11az-2022.
    case 40
        numNz = 242; % Number of non-zero subcarriers as defined in Equation 27-126b of IEEE Std 802.11az-2022.
    case 80
        numNz = 498; % Number of non-zero subcarriers as defined in Equation 27-126c of IEEE Std 802.11az-2022.
    otherwise % 160 MHz
        numNz = 996; % Number of non-zero subcarriers as defined in Figure 27-46g of IEEE Std 802.11az-2022.
end
numUsers = numel(cfg.User);
numRepetitionsPerUser = coder.nullcopy(zeros(1,numUsers));
numSTSPerUser = coder.nullcopy(zeros(1,numUsers));
numHELTFSymPerUserWithRep = coder.nullcopy(zeros(1,numUsers));
numHELTFSymPerUserWithoutRep = coder.nullcopy(zeros(1,numUsers));
for u=1:numUsers
    numSTSPerUser(u) = cfg.User{u}.NumSpaceTimeStreams;
    numRepetitionsPerUser(u) = cfg.User{u}.NumHELTFRepetitions;
    numHELTFSymPerUserWithoutRep(u) = wlan.internal.numVHTLTFSymbols(cfg.User{u}.NumSpaceTimeStreams);
    numHELTFSymPerUserWithRep(u) = numHELTFSymPerUserWithoutRep(u)*cfg.User{u}.NumHELTFRepetitions;
end
numHELTFTotal = sum(numHELTFSymPerUserWithRep);
numOctetsPerUser = numNz*numHELTFSymPerUserWithRep+7;  % First 7 octets are for phase rotation applied on number of space-time streams greater than 1
hezInfo = struct(...
    'NUsers',                     numUsers,...
    'NSTS',                       numSTSPerUser,...
    'MaxNSTS',                     max(numSTSPerUser),...
    'NRepetitions',               numRepetitionsPerUser,...
    'NHELTF',                     numHELTFTotal,...
    'NHELTFWithRepetition',       numHELTFSymPerUserWithRep,...
    'NHELTFWithoutRepetition',    numHELTFSymPerUserWithoutRep,...
    'NSecureHELTFOctets',         numOctetsPerUser);
end