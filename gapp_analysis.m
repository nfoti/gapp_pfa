
%%%%%%%%%%%%OLD SFI%%%%%%%%%%%%%%%%%
%OLD
% datapath = 'smp_sfi_psidict.05_6alphas.mat';
% 
% load(datapath);

%for BAD TOPICS! pre debug
%we use the 4th struct, K = 131
%
%8-chaos
%14-power law
%80-networks
%106-entropy
%22-quantum computing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%6th result for CONS dyn
%3 agriculture
%5 parliamnet/punishment
%9 kings etc ...

%6th result NIPS dyn
%2 reinforcement learning
%6,8 neural network?

%WIDER kernels for NIPS, run again; .0025 NIPS
%.005 SOTU CONS

par = result_params{6};

clearvars -except tstamps words par result_params

Phi = par.Phi_s;
Znk = par.Znk_s;
Psi = par.Psi_s;
psi_inds = par.psi_inds_s;
mus = par.mus;
W = par.W_s;
psi_dict = par.psi_dict;
Theta = par.Theta_s;

K = numel(Theta)

unq_yrs = unique(tstamps);

% smooth = numel(unq_yrs);
smooth = 250;
yrs = linspace(unq_yrs(1), unq_yrs(end), smooth);

Km =  computeKernMats_exp2(yrs', struct('Psi', Psi, 'mus', mus, 'psi_inds', psi_inds, 'psi_dict', psi_dict));


topic_pop = zeros(K, smooth);
for i=1:K
    topic_pop(i,:) = normcdf(Km{i} * W(:,i));
end

[topwords topprobs] = topic_top_N_words(Phi, 20, words);



%print words, show topic popularity plots
for i = 1:size(topic_pop,1)
clf;
plot(yrs, topic_pop(i,:)); axis([0 1 0 1]);
title(['Topic ' num2str(i)]);
topwords{i}
topprobs(:,i)
pause
end

