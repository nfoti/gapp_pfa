
%%%NIPS

load('NIPS_perplexity.mat');

%%%perp vs K plot
figure;
hold on;

title('Perplexity for Various Models on NIPS');
xlabel('K'); ylabel('perplexity'); 

plot(K_NIPS_BNBP, perp_NIPS_BNBP, 'b');
plot(K_NIPS_NMF, perp_NIPS_NMF, 'g');
plot(K_NIPS_dyn, perp_NIPS_dyn, 'r');
plot(K_NIPS_static, perp_NIPS_static, 'k');
grid on;

legend('BNBP', 'NMF', 'dynamic', 'static');

%%%perp vs alpha plot
figure;
hold on; grid on;

title('Perplexity for Various Models on NIPS');
xlabel('\alpha'); ylabel('perplexity'); 

plot(aphi, perp_NIPS_BNBP, 'b');
plot(aphi, perp_NIPS_dyn, 'r');
plot(aphi, perp_NIPS_static, 'k');
grid on;

legend('BNBP', 'dynamic', 'static');

clear;

%%%CONS

load('CONS_perplexity.mat');

%%%perp vs K plot
figure;
hold on; 

title('Perplexity for Various Models on CONS');
xlabel('K'); ylabel('perplexity'); 

plot(K_CONS_BNBP, perp_CONS_BNBP, 'b');
plot(K_CONS_NMF, perp_CONS_NMF, 'g');
plot(K_CONS_dyn, perp_CONS_dyn, 'r');
plot(K_CONS_static, perp_CONS_static, 'k');
grid on;

legend('BNBP', 'NMF', 'dynamic', 'static');

%%%perp vs alpha plot
figure;
hold on; grid on;

title('Perplexity for Various Models on CONS');
xlabel('\alpha'); ylabel('perplexity'); 

plot(aphi, perp_CONS_BNBP, 'b');
plot(aphi, perp_CONS_dyn, 'r');
plot(aphi, perp_CONS_static, 'k');
grid on;

legend('BNBP', 'dynamic', 'static');

clear;
