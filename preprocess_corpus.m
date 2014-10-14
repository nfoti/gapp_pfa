function [counts, words, years] = preprocess_corpus(counts, words, years, ...
    q, docthresh, wordthresh)
%function preprocesses set of data.
%
%specifically: 
%1. kill stop words 
%2. kill off bad docs/words (ie under a threshold/not at all)
%3. kill off bad years
%4. tf-idf prune
%
%q = 0.3-0.4 is good range for SFI
%
%Inputs: sparse counts matrix
%cell array of words
%vector of years
%percent q to keep via tf-idf (throw away 1-q percent)
%
%Returns: processed sparse counts matrix
%pruned words cell array
%pruned years vector

fid = fopen('./stopwordlist.txt');
stopwords = cell(1,456);

tline = fgetl(fid);
i = 1;
while ischar(tline)
    stopwords(i) = {tline};
    tline = fgetl(fid);
    i = i+1;
end
fclose(fid);
clear fid tline i

%kill stopwords
inds = true(numel(words),1);
for i=1:numel(stopwords)
    inds(strcmp(stopwords{i},words)) = false;
end
words = words(inds);
counts = counts(inds,:);
clear inds


%Filter out documents that have < 50 words total
%Filter out words that appear less than 5 times in all of corpus
min_doc_ct = docthresh;
min_word_ct = wordthresh;

baddocs = find(sum(counts)<min_doc_ct);
counts(:,baddocs) = [];
years(baddocs) = [];
badwords = find(sum(counts,2)<=min_word_ct);
counts(badwords,:) = [];
words(badwords) = [];
clear baddocs badwords min_doc_ct min_word_ct;

%kill off years with too few docs (eg 1989, 2012 for SFI)
y = unique(years);
thresh = 5;
for i=1:length(y)
    badyears = years==y(i);
    if sum(badyears) < thresh
        years(badyears) = [];
        counts(:,badyears) = [];
    end
end
clear y thresh badyears i;


% tf-idf of each word
tfidf = sum(counts,2) .* log(size(counts,2)./(sum(counts>0,2))+1);
tfidf = full(tfidf);

% Keep top q% of words w.r.t tfidf
Tw  = quantile(tfidf,1-q+eps);
wind = tfidf > Tw;

words = words(wind);
counts = counts(wind,:);
clear tidf Tw wind q;

end

