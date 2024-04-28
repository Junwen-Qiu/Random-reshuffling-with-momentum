%
% Nonconvex binary classification with L2 regularization.
%
% Functions:
% @obj   computes the function value
% @grad  compute the (stochastic) gradient

function f_handle  = f_loss(L,n)
    f_handle.obj   = @obj;
    f_handle.grad  = @grad;
    lambda         = L/sqrt(n);

    function f_val = obj(AT,b,x)
        y          = exp(2*b.*(x'*AT)');
        f_val      = 1-mean((y-1)./(y+1)) + lambda*norm(x)^2/2;
    end

    function gf    = grad(AT,b,x)
         y         = exp(2*b.*(x'*AT)');
         cache     = (y-1)./(y+1);
         gf        = mean(-(b.*(1-cache.^2))'.*AT,2) + lambda*x; 
    end
end