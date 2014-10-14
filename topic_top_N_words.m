function [top_words word_probs] = topic_top_N_words(Phi, N, words)
%topic_top_N_words
%
%Inputs: Matrix Phi of Distributions over words for topics, size P x K where
%P is size of vocab, K number of topics
%words the full cell array of vocab words
%N number of top words to see per topic
%
%Outputs: Cell array of size N x K with the top words per topic in each of
%the K cols
%N x K matrix giving probability of observing the word with same position
%in the cell array top_words

[P K] = size(Phi);
top_words = cell(1,K);
word_probs = zeros(N,K);

for k=1:K
    topic = Phi(:,k);
    [~,inds] = sort(topic, 'descend');
    indsN = inds(1:N);
    top_words{k} = words(indsN);
    word_probs(:,k) = topic(indsN);
end
