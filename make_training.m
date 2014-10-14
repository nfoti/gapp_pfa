function [train_cts,test_cts] = make_training(counts, x)
%MAKE_TRAINING convert data set into training and 
%
%Inputs: 
%Sparse counts matrix
%percent x to make into test set (train on 1-x percent)
%
%Returns: 
%sparse counts matrix - training set, custom format
%sparse counts matrix - test set, custom format

[P,N] = size(counts);

doc_sizes = sum(counts); %total words per doc
tw_perdoc = floor(x*doc_sizes); %each elt is size of test set for that doc
test_size = sum(tw_perdoc); %total test size

test_inds = zeros(test_size, 2); %store test set [wordID docID]
idx = 1;

%loop over docs
for n=1:N
    ii = find(counts(:,n)); %find words
    words = zeros(doc_sizes(n),1); %enumerate words for doc
    rsum = 1;
    %add as many indices for this word as appropriate acc to freq
    for j = 1:length(ii) %loop over words in doc
        p = ii(j); %individual word index
        words(rsum:(rsum+counts(p,n)-1)) = p;
        rsum = rsum + counts(p,n);
    end
    rp = randperm(doc_sizes(n));
    
    %fill in test words for this doc
    test_inds(idx:(idx+tw_perdoc(n)-1),:) = ...
        [words(rp(1:tw_perdoc(n))), n*ones(tw_perdoc(n),1)];
    
    idx = idx + tw_perdoc(n);
    
end

test_sparse = sparse(test_inds(:,1), test_inds(:,2), 1, P, N);
[ii,jj,s] = find(test_sparse);
test_cts = struct;
test_cts.inds = [ii jj];
test_cts.vals = s;
test_cts.P = size(test_sparse,1);
test_cts.N = size(test_sparse,2);
test_cts.sum = sum(test_sparse(:));


train_sparse = counts - test_sparse;
[ii,jj,s] = find(train_sparse);
train_cts = struct;
train_cts.inds = [ii jj];
train_cts.vals = s;
train_cts.P = size(train_sparse,1);
train_cts.N = size(train_sparse,2);
train_cts.sum = sum(train_sparse(:));

end

