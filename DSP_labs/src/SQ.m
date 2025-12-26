clear;
close all;
clc;

% Load and preprocess image
originalImg = imread('/MATLAB Drive/image.jpg');
%originalGray = rgb2gray(originalImg);
processingImg = double(originalImg);
maxVal = 255;

% Define bit depths
bits = [1, 2, 3, 4];

% Initialize cell arrays to store quantized images
quantizedUniform = cell(length(bits), 1);
quantizedLaplacian = cell(length(bits), 1);
quantizedGaussian = cell(length(bits), 1);

% Uniform Quantization
for i = 1:length(bits)
    nLevels = 2^bits(i);
    stepSize = maxVal / (nLevels - 1);
    quantizedUniform{i} = uint8(round(processingImg / stepSize) * stepSize);
end

% Non-Uniform Laplacian Quantization
for i = 1:length(bits)
    nLevels = 2^bits(i);
    if nLevels == 2
        threshold = maxVal / 2;
        quantized = zeros(size(processingImg));
        quantized(processingImg >= threshold) = maxVal;
        quantizedLaplacian{i} = uint8(quantized);
    else
        b = maxVal / 4;
        mu = maxVal / 2;
        p = linspace(0, 1, nLevels + 1);
        laplacian_levels = zeros(1, length(p));
        for j = 1:length(p)
            if p(j) <= 0.5
                laplacian_levels(j) = mu - b * log(2 * (1 - p(j)));
            else
                laplacian_levels(j) = mu + b * log(2 * p(j));
            end
        end
        laplacian_levels = sort(laplacian_levels);
        laplacian_levels = laplacian_levels(2:end - 1);
        laplacian_levels = max(0, min(maxVal, laplacian_levels));
        quantized = zeros(size(processingImg));
        mask = processingImg < laplacian_levels(1);
        quantized(mask) = 0;
        for k = 1:length(laplacian_levels) - 1
            mask = processingImg >= laplacian_levels(k) & processingImg < laplacian_levels(k + 1);
            quantized(mask) = round((laplacian_levels(k) + laplacian_levels(k + 1)) / 2);
        end
        mask = processingImg >= laplacian_levels(end);
        quantized(mask) = maxVal;
        quantizedLaplacian{i} = uint8(quantized);
    end
end

% Non-Uniform Gaussian Quantization
for i = 1:length(bits)
    nLevels = 2^bits(i);
    sigma = 0.5; % Fixed sigma as per original code
    if nLevels == 2
        threshold = maxVal / 2;
        quantized = zeros(size(processingImg));
        quantized(processingImg >= threshold) = maxVal;
        quantizedGaussian{i} = uint8(quantized);
    else
        gaussian_levels = norminv(linspace(0, 1, nLevels + 1), 0.5, sigma);
        gaussian_levels = gaussian_levels(2:end - 1);
        gaussian_levels = gaussian_levels * maxVal;
        gaussian_levels = max(0, min(maxVal, gaussian_levels)); % Ensure levels are within bounds
        quantized = zeros(size(processingImg));
        mask = processingImg < gaussian_levels(1);
        quantized(mask) = 0;
        for k = 1:length(gaussian_levels) - 1
            mask = processingImg >= gaussian_levels(k) & processingImg < gaussian_levels(k + 1);
            quantized(mask) = round((gaussian_levels(k) + gaussian_levels(k + 1)) / 2);
        end
        mask = processingImg >= gaussian_levels(end);
        quantized(mask) = maxVal;
        quantizedGaussian{i} = uint8(quantized);
    end
end

% Create figure for displaying images
figure('Name', 'Scalar Quantization Results', 'Position', [100 100 1200 800]);

% Display original image
subplot(3, 5, 1);
imshow(originalImg);
title('Original', 'FontSize', 10);

% Display quantized images
for i = 1:length(bits)
    % Uniform
    subplot(3, 5, 1 + 3 * (i - 1) + 1);
    imshow(quantizedUniform{i});
    title(sprintf('%d-bit Uniform', bits(i)), 'FontSize', 10);
    
    % Laplacian
    subplot(3, 5, 1 + 3 * (i - 1) + 2);
    imshow(quantizedLaplacian{i});
    title(sprintf('%d-bit Laplacian', bits(i)), 'FontSize', 10);
    
    % Gaussian
    subplot(3, 5, 1 + 3 * (i - 1) + 3);
    imshow(quantizedGaussian{i});
    title(sprintf('%d-bit Gaussian', bits(i)), 'FontSize', 10);
end

% Compute and print metrics
fprintf('Quantization Metrics:\n');
for i = 1:length(bits)
    % Uniform
    mse = mean((processingImg(:) - double(quantizedUniform{i}(:))).^2);
    psnr = 10 * log10(maxVal^2 / mse);
    compression_ratio = 8 / bits(i);
    fprintf('%d-bit Uniform: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', bits(i), psnr, compression_ratio);
    
    % Laplacian
    mse = mean((processingImg(:) - double(quantizedLaplacian{i}(:))).^2);
    psnr = 10 * log10(maxVal^2 / mse);
    fprintf('%d-bit Laplacian: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', bits(i), psnr, compression_ratio);
    
    % Gaussian
    mse = mean((processingImg(:) - double(quantizedGaussian{i}(:))).^2);
    psnr = 10 * log10(maxVal^2 / mse);
    fprintf('%d-bit Gaussian: PSNR = %.2f dB, Compression Ratio = %.2f:1\n', bits(i), psnr, compression_ratio);
end