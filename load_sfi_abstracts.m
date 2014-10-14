function [X T vocab] = load_sfi_abstracts(prefix)
% LOAD_SFI_ABSTRACTS Load SFI abstracts data set from preprocessed data.
%
% [X T vocab] = load_sfi_abstracts(prefix)
%
% Inputs:
%   prefix : path to prefix of files to load, extensions will be appended
%   
% Returns:
%   X : PxN sparse matrix of word counts
%   T : Nx1 vector of years for each abstract
%   vocab : 1xP cell array of words that make up the vocab

% Could do this with importdata also
f = fopen([prefix '.counts'], 'r');
C = textscan(f, '%f,%f,%f', 'Headerlines', 1);
fclose(f);

vocab = importdata([prefix '.vocab']);
P = numel(vocab);

T = importdata([prefix '.years']);
N = numel(T);

X = sparse(C{1}+1, C{2}+1, double(C{3}), P, N);