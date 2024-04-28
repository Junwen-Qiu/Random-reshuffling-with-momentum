clearvars -except AT b L;
[n,N]        = size(AT);
trial        = 10; 
x0           = zeros(n,1); 
cand_alpha   = 1;
opt.alpha    = cand_alpha;
opt.beta     = 0.9;
opt.maxit    = 100;
opt.batch    = 512; 
opt.L        = L;
opt.prt      = true;
opt.gamma    = 1; 

objf         = f_loss(L,N);

parfor k = 1:trial
    [x_rr,out_rr]      = StoAlg(objf,AT,b,x0,opt,'RR'); 
    rr_fval_hist(:,k)  = out_rr.fval;
    rr_x_hist(:,k)     = x_rr;

    [x_rrm,out_rrm]    = StoAlg(objf,AT,b,x0,opt,'RRM'); 
    rrm_fval_hist(:,k) = out_rrm.fval;
    rrm_x_hist(:,k)    = x_rrm;

    [x_som,out_som]    = StoAlg(objf,AT,b,x0,opt,'SOM'); 
    som_fval_hist(:,k) = out_som.fval;
    som_x_hist(:,k)    = x_som;

    [x_sgd,out_sgd]    = StoAlg(objf,AT,b,x0,opt,'SGD'); 
    sgd_fval_hist(:,k) = out_sgd.fval;
    sgd_x_hist(:,k)    = x_sgd;

    [x_sgm,out_sgm]    = StoAlg(objf,AT,b,x0,opt,'SGM'); 
    sgm_fval_hist(:,k) = out_sgm.fval;
    sgm_x_hist(:,k)    = x_sgm;
    
end  
    [x_igm,out_igm]    = StoAlg(objf,AT,b,x0,opt,'IGM'); 
    igm_fval_hist      = out_igm.fval;
    igm_x_hist         = x_igm;