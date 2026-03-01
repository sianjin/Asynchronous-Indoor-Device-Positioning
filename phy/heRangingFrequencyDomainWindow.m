function fdw = heRangingFrequencyDomainWindow(Nfft,kNz)
%heRangingFrequencyDomainWindow HE Ranging Frequency Domain Window
%   fdw = heRangingFrequencyDomainWindow(Nfft,kNz) generates the frequency
%   domain windowing samples as defined in 802.11az-2022, Equation 27-126g.
%
%   fdw are the samples of frequency domain window function.It is a complex
%   vector of size Nfft-by-1 where Nfft represents the FFT size.
%
%   kNz are the indices of non-zero subcarriers for the given a given
%   bandwidth. It is a vector of size numNz-by-1 where numNz are the number
%   of non-zero subcarriers.

%   Copyright 2023 The MathWorks, Inc.

%#codegen

assert(mod(Nfft,2)==0,"Nfft size must be an even number")
n = -10:10;
Nfd = 2*Nfft; % Equation 27-126h
k = -Nfft/2:Nfft/2-1;
pn = flattopwin(numel(n)); % Equatation 27-126i
wfd = sum(pn'.*exp(-1i*pi*k'*n/Nfd),2); % Equation 27-126g
wfdRMS = rms(wfd(kNz+Nfft/2+1));
fdw = wfd./wfdRMS; % Normalized to have unit RMS power
end