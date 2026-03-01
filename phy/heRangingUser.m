classdef heRangingUser < comm.internal.ConfigBase
%heRangingUser User properties of HE-LTF field
%   CFGUser = heRangingUser creates a user configuration object. This
%   object contains the HE-LTF properties of a user within an HE Ranging
%   NDP PPDU.
%
%   CFGUSER = heRangingUser(Name,Value) creates an object that holds
%   the HE-LTF properties for the users within an HE Ranging NDP PPDU,
%   CFGUSER, with the specified property Name set to the specified value.
%   You can specify additional name-value pair arguments in any order as
%   (Name1,Value1, ...,NameN,ValueN).
%
%   heRangingUser objects are used to parameterize users within an HE
%   Ranging NDP PPDU transmission, and therefore are part of the
%   <a href="matlab:help('heRangingConfig')">heRangingConfig</a> object.
%
%   heRangingUser properties:
%
%   NumSpaceTimeStreams - Number of space-time streams
%   NumHELTFRepetitions - Number of repetitions of HE-LTF symbols
%   SecureHELTFOctets   - Secure HE-LTF octets
%
%   See also heRangingConfig.

%   Copyright 2020-2025 The MathWorks, Inc.

%#codegen

properties
    %NumSpaceTimeStreams Number of space-time streams
    %   Specify the number of space-time streams as integer between 1 and
    %   8, inclusive. The default value of this property is 1.
    NumSpaceTimeStreams (1,1) {mustBeNumeric,mustBeInteger,mustBeGreaterThanOrEqual(NumSpaceTimeStreams,1),mustBeLessThanOrEqual(NumSpaceTimeStreams,8)} = 1;
    %NumHELTFRepetitions Number of repetitions of HE-LTF symbols
    %   Specify the number of HE-LTF repetition as integer between 1 and 8,
    %   inclusive. The default value of this property is 2, which is the
    %   minimum number of repetitions of HE-LTF symbols in HE-LTF secure
    %   mode.
    NumHELTFRepetitions (1,1) {mustBeNumeric,mustBeInteger,mustBeGreaterThanOrEqual(NumHELTFRepetitions,1),mustBeLessThanOrEqual(NumHELTFRepetitions,8)} = 2;
    %SecureHELTFOctets Secure HE-LTF octets
    %   Specify the secure HE-LTF octets as a character vector or string
    %   scalar representing octets in hexadecimal format or a uint8 vector
    %   or a numeric vector whose elements are within the range [0,255].
    %   If the number of input octets is less than the required number of
    %   octets for the given user configuration, the secure octets are
    %   cyclically extended. If the number of input octets is more than the
    %   required number of octets for the given user configuration, only the
    %   required number of octets are extracted from the input.
    %   The default sequence is '00000000'.
    SecureHELTFOctets = '00000000';
end

methods
    function obj = heRangingUser(varargin)
        % For codegen set different dimensions to make sure varsize
        if ~isempty(coder.target)
            secureHELTFOctets = '00000000';
            coder.varsize('SecureHELTFOctets',[1 63751*2],[0 1]); % Add variable-size support,
            % 63751=64*996+7, where 64 is max number of HE-LTF symbols,
            % 996 is number of non-zero subcarriers in 160 MHz bandwidth
            % and 7 is the number of octets required to add phase rotation
            % for space-times streams 2 to 7. The number is multiplied by 2
            % since each octet requires 2 hex digits.
            obj.SecureHELTFOctets = secureHELTFOctets; % Default
        end
        obj = setProperties(obj,varargin{:}); % Supperclass method for NV pair parsing
    end

    function obj = set.SecureHELTFOctets(obj,val)
        propName = 'SecureHELTFOctets';
        % validate format
        validateattributes(val, {'numeric','char', 'string'}, {}, mfilename, propName);
        if isa(val,'numeric')
            validateattributes(val,{'numeric'},{'finite','nonempty','>=',0,'<=',255,'integer','vector'})
            obj.(propName) = val;
        else % char or string
            if isa(val, 'char')
                validateattributes(val,{'char'},{'vector'},mfilename,propName);
            else % string
                validateattributes(val,{'string'},{'scalar'},mfilename,propName);
            end
            % Validate hex-digits
            wnet.internal.validateHexOctets(upper(char(val)),propName);
            obj.(propName) = val;
        end
    end
end
end
