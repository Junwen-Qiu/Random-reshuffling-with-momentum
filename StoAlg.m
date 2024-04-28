function [x,out] = StoAlg(f,AT,b,x0,opt,alg)
    x            = x0;
    [n,m]        = size(AT);
    num_inner    = floor(m./opt.batch);
    rem          = mod(m,opt.batch);
    dx           = zeros(n,1);
    beta         = opt.beta;
    fval         = zeros(opt.maxit,1);
    switch alg
        case "IGM"
            idx  = 1:m;
        case "SOM"
            idx  = randperm(m);
        case {"RR","SGD"}
            beta = 0;
    end

    for t = 1:opt.maxit
        switch alg
            case {"RRM","RR"}
                idx = randperm(m);
            case {"SGM","SGD"}
                idx = randi(m,1,m);
        end
        fval(t)     = f.obj(AT,b,x);
        if opt.prt; fprintf("%s: beta = %4.2f Iter %3d  Fval %4.4e \n",alg,beta,t,fval(t)); end

        alpha       = opt.alpha/(opt.L*t^opt.gamma);

        for k = 0:num_inner-1
            blk_idx = idx(k*opt.batch+1:(k+1)*opt.batch);
            fgrad   = f.grad(AT(:,blk_idx),b(blk_idx),x);
            dx      = - alpha*fgrad + beta*dx;
            x       = x + dx;
        end
        if min(rem,k)>0 
            blk_idx = idx(num_inner*opt.batch:m);
            fgrad   = f.grad(AT(:,blk_idx),b(blk_idx),x);
            dx      = - alpha*fgrad + beta*dx;
            x       = x + dx;
        end
    end
    out.fval        = fval;
end

