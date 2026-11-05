function purity = Purity(Ltrue, Lpred)
%Purity Clustering purity (1 - best-match not required).
%   purity = 1/N * sum_k max_j |C_k ∩ T_j|

Ltrue = Ltrue(:);
Lpred = Lpred(:);
if numel(Ltrue) ~= numel(Lpred)
    error('size(Ltrue) must == size(Lpred)');
end

N = numel(Ltrue);
if N == 0
    purity = 0;
    return;
end

% Convert labels to consecutive indices
[~,~,t] = unique(Ltrue);
[~,~,p] = unique(Lpred);

nPred = max(p);
nTrue = max(t);

% Contingency table Nij: samples assigned to pred-cluster i, true-class j
Nij = accumarray([p, t], 1, [nPred, nTrue]);

% Purity: for each predicted cluster take the majority true class
purity = sum(max(Nij, [], 2)) / N;
end

