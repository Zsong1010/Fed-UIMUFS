function upload = SecureMaskUploadRAFA(aligned_W, phi, valid, local_obj, client_id, state, round_id)
%SECUREMASKUPLOADRAFA Mask RAFA sufficient statistics and local objective.
% Pairwise masks cancel only after the fixed client roster is summed.
K = state.K;
V = numel(aligned_W);
if client_id < 1 || client_id > K || client_id ~= floor(client_id)
    error('SecureMaskUploadRAFA:InvalidClientId', 'Invalid client identifier.');
end
if numel(phi) ~= V || numel(valid) ~= V || any(~isfinite(phi)) || ...
        any(phi < 0) || ~isscalar(local_obj) || ~isfinite(local_obj)
    error('SecureMaskUploadRAFA:InvalidStatistics', 'Invalid RAFA evidence or objective.');
end
if round_id < 1 || round_id ~= floor(round_id)
    error('SecureMaskUploadRAFA:InvalidRound', 'Round identifier must be a positive integer.');
end

shapes = cellfun(@size, aligned_W, 'UniformOutput', false);
payload = zeros(0, 1);
for v = 1:V
    w = double(aligned_W{v}(:));
    if any(~isfinite(w))
        error('SecureMaskUploadRAFA:NonfiniteSelector', 'Selectors must be finite.');
    end
    if logical(valid(v))
        payload = [payload; phi(v) .* w; phi(v); w; 1]; %#ok<AGROW>
    else
        payload = [payload; zeros(2*numel(w) + 2, 1)]; %#ok<AGROW>
    end
end
payload = [payload; double(local_obj)];

q = state.modulus;
scale = state.scale;
fixed = round(payload .* scale);
if any(abs(fixed) >= q / (2 * K))
    error('SecureMaskUploadRAFA:FixedPointOverflow', ...
        'Quantized payload exceeds the safe modular range; reduce scale or increase modulus.');
end
masked = mod(fixed, q);
for peer = 1:K
    if peer == client_id, continue; end
    pair_seed = rafa_pair_secret(state.private_keys{client_id}, state.public_keys{peer});
    lo = min(client_id, peer);
    hi = max(client_id, peer);
    context = sprintf('Fed-UIMUFS-RAFA-v1|%s|round=%d|pair=%d-%d|dim=%d', ...
        state.session_id, round_id, lo, hi, numel(payload));
    mask = rafa_hmac_stream(pair_seed, context, numel(payload), q);
    if client_id < peer
        masked = mod(masked + mask, q);
    else
        masked = mod(masked - mask, q);
    end
end

upload.values = masked;
upload.shapes = shapes;
upload.dimension = numel(payload);
upload.client_id = client_id;
upload.round_id = round_id;
upload.scale = scale;
upload.modulus = q;
end

function secret = rafa_pair_secret(private_key, peer_public_b64)
peer_bytes = java.util.Base64.getDecoder().decode(peer_public_b64);
key_factory = java.security.KeyFactory.getInstance('EC');
peer_public = key_factory.generatePublic(java.security.spec.X509EncodedKeySpec(peer_bytes));
agreement = javax.crypto.KeyAgreement.getInstance('ECDH');
agreement.init(private_key);
agreement.doPhase(peer_public, true);
secret = agreement.generateSecret();
end

function values = rafa_hmac_stream(secret, context, count, q)
mac = javax.crypto.Mac.getInstance('HmacSHA256');
mac.init(javax.crypto.spec.SecretKeySpec(secret, 'HmacSHA256'));
context_bytes = uint8(unicode2native(context, 'UTF-8'));
values = zeros(count, 1);
written = 0;
counter = uint64(0);
while written < count
    message = [context_bytes, uint8('|ctr='), uint8(sprintf('%u', counter))];
    digest = typecast(mac.doFinal(typecast(message, 'int8')), 'uint8');
    for offset = 1:7:28
        if written >= count, break; end
        bytes = reshape(double(digest(offset:offset + 6)), 1, []);
        value = sum(bytes(1:6) .* (256 .^ (0:5))) + mod(bytes(7), 8) * 2^48;
        values(written + 1) = mod(value, q);
        written = written + 1;
    end
    counter = counter + 1;
end
end
