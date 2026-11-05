function ari = ARI(Ltrue, Lpred)
%ARI Adjusted Rand Index.
%   Computed from contingency table using combination counts.

Ltrue = Ltrue(:);
Lpred = Lpred(:);
if numel(Ltrue) ~= numel(Lpred)
    error('size(Ltrue) must == size(Lpred)');
end

N = numel(Ltrue);
if N < 2
    ari = 0;
    return;
end

[~,~,t] = unique(Ltrue);
[~,~,p] = unique(Lpred);

nPred = max(p);
nTrue = max(t);
Nij = accumarray([p, t], 1, [nPred, nTrue]);

comb2 = @(x) x .* (x - 1) / 2;

Index = sum(comb2(Nij(:)));   % sum_ij C(n_ij,2)
a = sum(Nij, 2);             % row sums
b = sum(Nij, 1);             % column sums

sumA = sum(comb2(a));       % sum_i C(a_i,2)
sumB = sum(comb2(b));       % sum_j C(b_j,2)
totalComb = comb2(N);       % C(N,2)

ExpectedIndex = (sumA * sumB) / totalComb;
MaxIndex = 0.5 * (sumA + sumB);

den = MaxIndex - ExpectedIndex;
if den == 0
    ari = 0;
else
    ari = (Index - ExpectedIndex) / den;
end

ari = real(ari);
end

