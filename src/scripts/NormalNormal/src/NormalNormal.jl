module NormalNormal

using Distributions, StatsFuns

export draw_truth, draw_obs
export sigma_eff, analytic_lpdf, analytic_p_of_zero
export simulate_p

function draw_truth(mu_true, sigma_true, Nobs)
    if mu_true == 0 && sigma_true == 0
        zeros(Nobs)
    else
        rand(Normal(mu_true, sigma_true), Nobs)
    end
end

function draw_obs(x_true)
    nobs = length(x_true)
    sigma_obs = rand(LogNormal(log(1.0), 0.5), nobs)
    x_obs = rand.(Normal.(x_true, sigma_obs))

    (x_obs, sigma_obs)
end

function sigma_eff(sigma_obs)
    se = 1.0/sqrt(sum(1.0 ./ (sigma_obs .* sigma_obs)))
    sv = 1.0/sqrt(sum(1.0 ./ (2.0 .* sigma_obs .* sigma_obs .* sigma_obs .* sigma_obs)))

    (se, sv)
end

function analytic_lpdf(x_obs, sigma_obs, mu, v, smu, sv)
    lp = 0.0
    if smu > 0
        lp += logpdf(Normal(0.0, smu), mu)
    end
    if sv > 0
        lp += logpdf(Normal(0.0, sv), v) + log(2) # Half normal.
    end

    lp += sum(logpdf.(Normal.(mu, sqrt.(v .+ sigma_obs.*sigma_obs)), x_obs))
    lp
end

function analytic_p_of_zero(x_obs, sigma_obs, scale_factor)
    se, sv = sigma_eff(sigma_obs)

    scale_mu = scale_factor*se
    scale_v = scale_factor*sv

    mus = -10*se:0.05*se:10*se
    vs = 0.0:0.05*sv:10*sv
    logps = [analytic_lpdf(x_obs, sigma_obs, mu, v, scale_mu, scale_v) for mu in mus, v in vs]
    logp0 = analytic_lpdf(x_obs, sigma_obs, 0.0, 0.0, scale_mu, scale_v)

    log_above = logsumexp(logps[logps .> logp0])
    log_all = logsumexp(logps)

    exp(log_above - log_all)
end

function simulate_p(mu_true, sigma_true, nobs; scale_factor=-1.0)
    xt = draw_truth(mu_true, sigma_true, nobs)
    xo, so = draw_obs(xt)

    analytic_p_of_zero(xo, so, scale_factor)
end

end # module NormalNormal
