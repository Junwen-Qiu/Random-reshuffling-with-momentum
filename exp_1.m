clearvars -except AT b L;
[n,N]         = size(AT);
trial         = 10; 
x0            = zeros(n,1); 
num_alg       = 6;
maxit         = 100;
cand_alpha    = 1;

opt1.alpha    = cand_alpha;
opt1.maxit    = maxit;
opt1.batch    = 512; 
opt1.prt      = true;
opt1.gamma    = 1; 
opt1.L        = L;

arrayfun(@(x)assignin('base',join(['opt',num2str(x)]),opt1), 2:num_alg);
opt1.beta    = 0;
opt2.beta    = 0.1;
opt3.beta    = 0.3;
opt4.beta    = 0.5;
opt5.beta    = 0.7;
opt6.beta    = 0.9;

objf         = f_loss(L,N);

arrayfun(@(x)assignin('base',join(['rrm',num2str(x),'_fval_hist']), zeros(opt1.maxit,trial)), 1:num_alg);
arrayfun(@(x)assignin('base',join(['rrm',num2str(x),'_fnat_hist']), zeros(opt1.maxit,trial)), 1:num_alg);
arrayfun(@(x)assignin('base',join(['rrm',num2str(x),'x_hist']), zeros(n,trial)), 1:num_alg);

parfor k = 1:trial
    [x_rrm1,out_rrm1]    = StoAlg(objf,AT,b,x0,opt1,'RR'); 
    rrm1_fval_hist(:,k)  = out_rrm1.fval;
    rrm1_x_hist(:,k)     = x_rrm1;

    [x_rrm2,out_rrm2]    = StoAlg(objf,AT,b,x0,opt2,'RRM'); 
    rrm2_fval_hist(:,k)  = out_rrm2.fval;
    rrm2_x_hist(:,k)     = x_rrm2;
    
    [x_rrm3,out_rrm3]    = StoAlg(objf,AT,b,x0,opt3,'RRM'); 
    rrm3_fval_hist(:,k)  = out_rrm3.fval;
    rrm3_x_hist(:,k)     = x_rrm3;
    
    [x_rrm4,out_rrm4]    = StoAlg(objf,AT,b,x0,opt4,'RRM'); 
    rrm4_fval_hist(:,k)  = out_rrm4.fval;
    rrm4_x_hist(:,k)     = x_rrm4;
    
    [x_rrm5,out_rrm5]    = StoAlg(objf,AT,b,x0,opt5,'RRM'); 
    rrm5_fval_hist(:,k)  = out_rrm5.fval;
    rrm5_x_hist(:,k)     = x_rrm5;
    
    [x_rrm6,out_rrm6]    = StoAlg(objf,AT,b,x0,opt6,'RRM'); 
    rrm6_fval_hist(:,k)  = out_rrm6.fval;
    rrm6_x_hist(:,k)     = x_rrm6;
end  