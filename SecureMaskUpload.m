function upload = SecureMaskUpload(local_W, sample_count, client_id, state, round_id, local_obj)
%SECUREMASKUPLOAD Mask weighted model parameters and one local objective value.
% Pairwise masks are derived by ECDH and expanded with HMAC-SHA-256.
if nargin < 6
    local_obj = 0;
end
if client_id < 1 || client_id > state.K || client_id ~= floor(client_id)
    error('SecureMaskUpload:InvalidClientId', 'Invalid client identifier.');
end
if ~isscalar(sample_count) || ~isfinite(sample_count) || ...
        sample_count <= 0 || sample_count ~= floor(sample_count)
    error('SecureMaskUpload:InvalidSampleCount', 'Sample count must be a positive integer.');
end
if ~isscalar(round_id) || round_id < 1 || round_id ~= floor(round_id)
    error('SecureMaskUpload:InvalidRound', 'Round identifier must be a positive integer.');
end

shapes = cellfun(@size, local_W, 'UniformOutput', false);
flat = cellfun(@(x) double(x(:)), local_W, 'UniformOutput', false);
flat = vertcat(flat{:});
if any(~isfinite(flat)) || ~isscalar(local_obj) || ~isfinite(local_obj)
    error('SecureMaskUpload:NonfinitePayload', ...
        'Model weights and local objective must be finite.');
end

q = state.modulus;
scale = state.scale;
weighted = round(flat .* (scale * double(sample_count)));
objective_fixed = round(double(local_obj) * scale);
if any(abs(weighted) >= q / (2 * state.K)) || abs(objective_fixed) >= q / (2 * state.K)
    error('SecureMaskUpload:FixedPointOverflow', ...
        'Quantized payload exceeds the safe modular range; lower scale or increase modulus.');
end
payload = [weighted; objective_fixed];
masked = mod(payload, q);

for peer = 1:state.K
    if peer == client_id
        continue;
    end
    pair_seed = secureagg_pair_secret(state.private_keys{client_id}, ...
        state.public_keys{peer});
    lo = min(client_id, peer);
    hi = max(client_id, peer);
    context = sprintf('Fed-UIMUFS-SA-v1|%s|round=%d|pair=%d-%d|dim=%d', ...
        state.session_id, round_id, lo, hi, numel(payload));
    mask = secureagg_hmac_stream(pair_seed, context, numel(payload), q);
    if client_id < peer
        masked = mod(masked + mask, q);
    else
        masked = mod(masked - mask, q);
    end
end

upload.values = masked;
upload.shapes = shapes;
upload.model_dimension = numel(flat);
upload.dimension = numel(payload);
upload.client_id = client_id;
upload.round_id = round_id;
upload.sample_count = double(sample_count);
upload.scale = scale;
upload.modulus = q;
end

function secret = secureagg_pair_secret(private_key, peer_public_b64)
peer_bytes = java.util.Base64.getDecoder().decode(peer_public_b64);
key_factory = java.security.KeyFactory.getInstance('EC');
peer_public = key_factory.generatePublic( ...
    java.security.spec.X509EncodedKeySpec(peer_bytes));
agreement = javax.crypto.KeyAgreement.getInstance('ECDH');
agreement.init(private_key);
agreement.doPhase(peer_public, true);
secret = agreement.generateSecret();
end

function values = secureagg_hmac_stream(secret, context, count, q)
% Four exactly represented 51-bit pseudorandom values per HMAC block.
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
        if written >= count
            break;
        end
        bytes = reshape(double(digest(offset:offset + 6)), 1, []);
        value = sum(bytes(1:6) .* (256 .^ (0:5))) + mod(bytes(7), 8) * 2^48;
        values(written + 1) = mod(value, q);
        written = written + 1;
    end
    counter = counter + 1;
end
end
