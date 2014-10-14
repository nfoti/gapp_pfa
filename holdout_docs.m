function [train,test,Xtrain,Xtest] = holdout_docs(counts, p, X)
% HOLDOUT_DOCS Create training and testing sets by holdoing out whole
% documents.
%
% Inputs:
%   counts : sparse P x N matrix, words in rows docs in columns
%   p : fraction of docs to hold out
%   X : [optional] N x d matrix of d-dimensional document covariates.  If 
%       specified, p*100% of the documents at each covariate are held out 
%       so that the test set is uniform over the data set.  Do not set if 
%       a uniform sample over all docs is desired.
%
% Returns:
%   train : sparse P x Ntrain matrix
%   test : sparse P x Ntest matrix
%
% Returned if X is specified
%   Xtrain : covariates for training observations
%   Xtest : covariates for test observations
%

[~,N] = size(counts);

if nargin < 3
    X = ones(N, 1);
end

Xunq = unique(X,'rows');

test_inds = [];

for i = 1:numel(Xunq)
   
    x = Xunq(i,:);
    inds = find(X == x);
    nx = numel(inds);
    
    % This will be different for each covariate
    ntest = round(p*nx);
    
    rp = randperm(nx);
    
    test_inds = [test_inds, inds(rp(1:ntest))']; %#ok
end

test_inds = sort(test_inds);
train_inds = setdiff(1:N, test_inds);

train = counts(:,train_inds);
test = counts(:,test_inds);

if nargin == 3
    Xtrain = X(train_inds,:);
    Xtest = X(test_inds,:);
else
    Xtrain = [];
    Xtest = [];
end