function fscore = Fscore(Ltrue, Lpred)
%Fscore Pairwise F-score (pairwise precision/recall/F1).
%   This computes F1 over all sample pairs:
%     TP: same in both partitions
%     FP: same in pred but different in true
%     FN: same in true but different in pred

Ltrue = Ltrue(:);
Lpred = Lpred(:);
if numel(Ltrue) ~= numel(Lpred)
    error('size(Ltrue) must == size(Lpred)');
end

N = numel(Ltrue);
if N < 2
    fscore = 0;
    return;
end

[~,~,t] = unique(Ltrue);
[~,~,p] = unique(Lpred);

nPred = max(p);
nTrue = max(t);
Nij = accumarray([p, t], 1, [nPred, nTrue]);

comb2 = @(x) x .* (x - 1) / 2;

TP = sum(comb2(Nij(:)));

a = sum(Nij, 2); % predicted cluster sizes
b = sum(Nij, 1); % true class sizes

sumA = sum(comb2(a));
sumB = sum(comb2(b));

FP = sumA - TP;
FN = sumB - TP;

denP = TP + FP;
if denP == 0
    precision = 0;
else
    precision = TP / denP;
end

denR = TP + FN;
if denR == 0
    recall = 0;
else
    recall = TP / denR;
end

if precision + recall == 0
    fscore = 0;
else
    fscore = 2 * precision * recall / (precision + recall);
end
end

