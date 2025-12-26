clear; close all; clc;

% Load and preprocess image
Img = imread('/MATLAB Drive/image.jpg');
%grayScale = rgb2gray(Img);
final_img = double(Img);
maxVal = 255;

% Define VQ parameters
block_height = 4;
block_width = 4;
L = block_height * block_width; 
K = 256; % Codebook size
max_iter = 10; % LBG iterations
num_samples = 10000; % For synthetic data generation

% Extract 4x4 blocks from image
[blocks, num_blocks_h, num_blocks_w] = extract_blocks(final_img, block_height, block_width);

% Method 1: VQ with Image Blocks
training_data1 = blocks;
codebook1 = lbg_training(training_data1, K, max_iter);
reconstructed_img1 = quantize_and_reconstruct(blocks, codebook1, num_blocks_h, num_blocks_w, block_height, block_width);

% Method 2: VQ with Gaussian Data
[mu, sigma] = estimate_gaussian_params(blocks);
training_data2 = generate_gaussian_data(mu, sigma, L, num_samples);
codebook2 = lbg_training(training_data2, K, max_iter);
reconstructed_img2 = quantize_and_reconstruct(blocks, codebook2, num_blocks_h, num_blocks_w, block_height, block_width);

% Method 3: VQ with Laplacian Data
[mu, b] = estimate_laplacian_params(blocks);
training_data3 = generate_laplacian_data(mu, b, L, num_samples);
codebook3 = lbg_training(training_data3, K, max_iter);
reconstructed_img3 = quantize_and_reconstruct(blocks, codebook3, num_blocks_h, num_blocks_w, block_height, block_width);

% Compute metrics
[psnr1, compression_ratio] = compute_metrics(final_img, reconstructed_img1, L, K);
[psnr2, ~] = compute_metrics(final_img, reconstructed_img2, L, K);
[psnr3, ~] = compute_metrics(final_img, reconstructed_img3, L, K);

% Display results
figure('Name', 'VQ Results', 'Position', [100 100 1200 800]);
subplot(2, 2, 1); imshow(Img); title('Original');
subplot(2, 2, 2); imshow(reconstructed_img1); title('VQ with Image Blocks');
subplot(2, 2, 3); imshow(reconstructed_img2); title('VQ with Gaussian Data');
subplot(2, 2, 4); imshow(reconstructed_img3); title('VQ with Laplacian Data');

% Print metrics
fprintf('VQ with Image Blocks: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', psnr1, compression_ratio);
fprintf('VQ with Gaussian Data: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', psnr2, compression_ratio);
fprintf('VQ with Laplacian Data: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', psnr3, compression_ratio);

% Helper Functions
function [blocks, num_blocks_h, num_blocks_w] = extract_blocks(img, block_height, block_width)
    [height, width] = size(img);
    num_blocks_h = height / block_height;
    num_blocks_w = width / block_width;
    num_blocks = num_blocks_h * num_blocks_w;
    blocks = zeros(block_height * block_width, num_blocks);
    idx = 1;
    for i = 1:block_height:height
        for j = 1:block_width:width
            block = img(i:i+block_height-1, j:j+block_width-1);
            blocks(:, idx) = block(:);
            idx = idx + 1;
        end
    end
end

function codebook = lbg_training(training_data, K, max_iter)
    [L, N] = size(training_data);
    indices = randperm(N, K);
    codebook = training_data(:, indices);
    for iter = 1:max_iter
        dist = zeros(N, K);
        for k = 1:K
            diff = training_data - codebook(:, k);
            dist(:, k) = sum(diff .^ 2, 1)';
        end
        [~, assignments] = min(dist, [], 2);
        for k = 1:K
            assigned = training_data(:, assignments == k);
            if ~isempty(assigned)
                codebook(:, k) = sum(assigned, 2) / size(assigned, 2);
            else
                codebook(:, k) = training_data(:, randi(N));
            end
        end
    end
end

function reconstructed_img = quantize_and_reconstruct(blocks, codebook, num_blocks_h, num_blocks_w, block_height, block_width)
    [L, num_blocks] = size(blocks);
    quantized_indices = zeros(1, num_blocks);
    for n = 1:num_blocks
        block = blocks(:, n);
        dist = sum((codebook - block) .^ 2, 1);
        [~, quantized_indices(n)] = min(dist);
    end
    quantized_blocks = codebook(:, quantized_indices);
    reconstructed_img = zeros(block_height * num_blocks_h, block_width * num_blocks_w);
    idx = 1;
    for i = 1:block_height:size(reconstructed_img, 1)
        for j = 1:block_width:size(reconstructed_img, 2)
            block = reshape(quantized_blocks(:, idx), block_height, block_width);
            reconstructed_img(i:i+block_height-1, j:j+block_width-1) = block;
            idx = idx + 1;
        end
    end
    reconstructed_img = uint8(reconstructed_img);
end

function [mu, sigma] = estimate_gaussian_params(blocks)
    mu = mean(blocks, 2);
    sigma = std(blocks, 0, 2);
end

function training_data = generate_gaussian_data(mu, sigma, L, num_samples)
    training_data = zeros(L, num_samples);
    for i = 1:L
        training_data(i, :) = mu(i) + sigma(i) * randn(1, num_samples);
    end
end

function [mu, b] = estimate_laplacian_params(blocks)
    mu = median(blocks, 2);
    b = mean(abs(blocks - mu), 2);
end

function training_data = generate_laplacian_data(mu, b, L, num_samples)
    training_data = zeros(L, num_samples);
    for i = 1:L
        u = rand(1, num_samples) - 0.5;
        training_data(i, :) = mu(i) - b(i) * sign(u) .* log(1 - 2 * abs(u));
    end
end

function [psnr, compression_ratio] = compute_metrics(original, reconstructed, L, K)
    mse = mean((original(:) - double(reconstructed(:))) .^ 2);
    psnr = 10 * log10(255^2 / mse);
    bits_per_block = log2(K);
    bits_per_pixel = bits_per_block / L;
    compression_ratio = 8 / bits_per_pixel;
end