function state = SecureAggregationSetup(K)
%SECUREAGGREGATIONSETUP Create per-client ECDH keys for pairwise mask setup.
% Threat model: honest-but-curious server, no client-server collusion, fixed
% online roster. This MATLAB project simulates clients in one process. In
% deployment, isolate private keys and authenticate all exchanged public keys.
if ~isscalar(K) || K < 2 || K ~= floor(K)
    error('SecureAggregationSetup:InvalidClientCount', 'K must be an integer >= 2.');
end
if ~usejava('jvm')
    error('SecureAggregationSetup:JavaRequired', ...
        'Secure aggregation requires MATLAB with a Java virtual machine.');
end

keygen = java.security.KeyPairGenerator.getInstance('EC');
keygen.initialize(java.security.spec.ECGenParameterSpec('secp256r1'));
state.K = K;
state.private_keys = cell(1, K);
state.public_keys = cell(1, K);
for k = 1:K
    pair = keygen.generateKeyPair();
    state.private_keys{k} = pair.getPrivate();
    state.public_keys{k} = char(java.util.Base64.getEncoder().encodeToString( ...
        pair.getPublic().getEncoded()));
end

% A public random session identifier prevents reuse of masks across sessions.
sr = java.security.SecureRandom();
session_bytes = zeros(1, 16, 'int8');
sr.nextBytes(session_bytes);
state.session_id = lower(reshape(dec2hex(typecast(session_bytes, 'uint8'), 2).', 1, []));
state.scale = 1e6;
state.modulus = 2^51 - 1; % Exact integer arithmetic is retained in MATLAB doubles.
end
