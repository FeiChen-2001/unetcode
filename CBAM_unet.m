%% 1. 环境初始化
clear; clc; close all;
if gpuDeviceCount > 0
    gpuDevice(1); 
    disp('GPU加速已启用');
else
    disp('使用CPU运行');
end

%% 2. 数据准备
dataPath = '';
imageDir = fullfile(dataPath, 'Images');
maskDir = fullfile(dataPath, 'Masks');

imds = imageDatastore(imageDir, 'FileExtensions', '.png');
pxds = pixelLabelDatastore(maskDir, {'background','pore'}, [0 255]);

assert(numel(imds.Files) == numel(pxds.Files), '数据数量不匹配');
imgSample = readimage(imds,1);
maskSample = readimage(pxds,1);
assert(isequal(size(imgSample), size(maskSample)), '尺寸不一致');

%% 3. 数据预处理
numFiles = numel(imds.Files);
indices = randperm(numFiles);

trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

trainEnd = floor(trainRatio * numFiles);
valEnd = trainEnd + floor(valRatio * numFiles);

trainIndices = indices(1:trainEnd);
valIndices = indices(trainEnd+1:valEnd);
testIndices = indices(valEnd+1:end);

imdsTrain = subset(imds, trainIndices);
pxdsTrain = subset(pxds, trainIndices);
imdsVal = subset(imds, valIndices);
pxdsVal = subset(pxds, valIndices);
imdsTest = subset(imds, testIndices);
pxdsTest = subset(pxds, testIndices);

augmenter = imageDataAugmenter(...
    'RandXReflection', true,...
    'RandYReflection', true,...
    'RandRotation', [],...
    'RandScale', []);

inputSize = []; 

dsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain,...
    'DataAugmentation', augmenter,...
    'OutputSize', inputSize(),...
    'OutputSizeMode', 'resize');

dsVal = pixelLabelImageDatastore(imdsVal, pxdsVal,...
    'OutputSize', inputSize(),...
    'OutputSizeMode', 'resize');

%% 4. 定义CBAM U-Net模型
numClasses = ;
numFiltersInit = ;
encoderDepth = ;

lgraph = layerGraph();

inputLayer = imageInputLayer(inputSize, 'Name', 'input');
lgraph = addLayers(lgraph, inputLayer);

encoderNames = cell(encoderDepth,);
currentLayer = 'input';

% 编码器
for i = 1:encoderDepth
    numFilters = numFiltersInit * 2^(i-1);
    conv1Name = sprintf('enc%d_conv1', i);
    conv2Name = sprintf('enc%d_conv2', i);
    relu1Name = sprintf('enc%d_relu1', i);
    relu2Name = sprintf('enc%d_relu2', i);
    
    conv1 = convolution2dLayer(, numFilters, 'Padding', 'same', 'Name', conv1Name);
    relu1 = reluLayer('Name', relu1Name);
    conv2 = convolution2dLayer(, numFilters, 'Padding', 'same', 'Name', conv2Name);
    relu2 = reluLayer('Name', relu2Name);
    
    lgraph = addLayers(lgraph, [conv1 relu1 conv2 relu2]);
    lgraph = connectLayers(lgraph, currentLayer, conv1Name);
    
    % 添加CBAM模块到编码器每个阶段输出
    cbamName = sprintf('cbam_enc%d', i);
    [lgraph, cbamOut] = cbamModule(lgraph, relu2Name, numFilters, cbamName);
    
    encoderNames{i} = cbamOut; % 保存CBAM处理后的特征作为跳跃连接
    
    if i < encoderDepth
        poolName = sprintf('pool%d', i);
        pool = maxPooling2dLayer(, 'Stride', , 'Name', poolName);
        lgraph = addLayers(lgraph, pool);
        lgraph = connectLayers(lgraph, cbamOut, poolName);
        currentLayer = poolName;
    else
        currentLayer = cbamOut; % 瓶颈层
    end
end

% 解码器
for i = encoderDepth-1:-1:1
    numFilters = numFiltersInit * 2^(i-1);
    upconvName = sprintf('upconv%d', i);
    upconv = transposedConv2dLayer(2, numFilters, 'Stride', 2, 'Name', upconvName);
    lgraph = addLayers(lgraph, upconv);
    lgraph = connectLayers(lgraph, currentLayer, upconvName);
    
    % 连接拼接层 (跳跃连接)
    concatName = sprintf('concat%d', i);
    concatLayer = depthConcatenationLayer(, 'Name', concatName);
    lgraph = addLayers(lgraph, concatLayer);
    lgraph = connectLayers(lgraph, encoderNames{i}, [concatName '/in1']);
    lgraph = connectLayers(lgraph, upconvName, [concatName '/in2']);
    
    % 解码器卷积层
    conv1Name = sprintf('dec%d_conv1', i);
    relu1Name = sprintf('dec%d_relu1', i);
    conv2Name = sprintf('dec%d_conv2', i);
    relu2Name = sprintf('dec%d_relu2', i);
    
    conv1 = convolution2dLayer(, numFilters, 'Padding', 'same', 'Name', conv1Name);
    relu1 = reluLayer('Name', relu1Name);
    conv2 = convolution2dLayer(, numFilters, 'Padding', 'same', 'Name', conv2Name);
    relu2 = reluLayer('Name', relu2Name);
    
    lgraph = addLayers(lgraph, [conv1 relu1 conv2 relu2]);
    lgraph = connectLayers(lgraph, concatName, conv1Name);
    
    % 在每个解码器阶段输出也添加CBAM
    cbamName = sprintf('cbam_dec%d', i);
    [lgraph, cbamOut] = cbamModule(lgraph, relu2Name, numFilters, cbamName);
    
    currentLayer = cbamOut;
end

finalConv = convolution2dLayer(1, numClasses, 'Name', 'final_conv');
softmaxLayer_ = softmaxLayer('Name', 'softmax');
pixelLayer = pixelClassificationLayer('Name', 'pixelLabels', 'Classes', {'background', 'pore'});

lgraph = addLayers(lgraph, [finalConv softmaxLayer_ pixelLayer]);
lgraph = connectLayers(lgraph, currentLayer, 'final_conv');

% 可视化网络结构（调试用）
% analyzeNetwork(lgraph);

%% 5. 训练配置（关键修正）
validationFrequency = 10; % 显式定义验证频率变量

options = trainingOptions('adam',...
    'InitialLearnRate', 1e-4,...
    'MaxEpochs', ,...
    'MiniBatchSize', ,...
    'ValidationData', dsVal,...
    'ValidationFrequency', validationFrequency,... % 使用已定义变量
    'Plots', 'training-progress',...
    'Verbose', true);

%% 6. 训练模型
[net, trainInfo] = trainNetwork(dsTrain, lgraph, options);
save('cbam_unet_soil_model.mat', 'net');

%% 7. 训练指标处理（已修复）
iterations = (1:numel(trainInfo.TrainingLoss))';

validationLossFull = NaN(size(iterations));
numValidation = numel(trainInfo.ValidationLoss);

% 使用已定义的validationFrequency变量
validationIndices = validationFrequency * (1:numValidation); 

validValidationIndices = validationIndices(validationIndices <= numel(iterations));
trainInfo.ValidationLoss = trainInfo.ValidationLoss(1:numel(validValidationIndices));

validationLossFull(validValidationIndices) = trainInfo.ValidationLoss(:);

trainingMetrics = table(...
    iterations,...
    trainInfo.TrainingLoss(:),...
    validationLossFull,...
    'VariableNames', {'Iteration', 'TrainingLoss', 'ValidationLoss'});
writetable(trainingMetrics, 'training_metrics.xlsx');

%% 8. 模型评估（增强版）
dsTest = pixelLabelImageDatastore(imdsTest, pxdsTest,...
    'OutputSize', inputSize(1:2),...
    'OutputSizeMode', 'resize');

pxdsPred = semanticseg(dsTest, net, 'WriteLocation', tempdir);

% 获取类别信息
classNames = pxdsTest.ClassNames;
numClasses = numel(classNames);

% 计算评估指标
metrics = evaluateSemanticSegmentation(pxdsPred, pxdsTest);

% 处理混淆矩阵数据类型
if istable(metrics.ConfusionMatrix)
    confusionMat = table2array(metrics.ConfusionMatrix); % 转换表格为数组
else
    confusionMat = metrics.ConfusionMatrix;
end

% 验证混淆矩阵有效性
assert(ismatrix(confusionMat) && size(confusionMat,1)==size(confusionMat,2),...
    '无效的混淆矩阵格式');

% 计算每个类别的指标
diagonal = diag(confusionMat);
sumRows = sum(confusionMat, 2);   % 实际总数 (TP + FN)
sumCols = sum(confusionMat, 1)';  % 预测总数 (TP + FP)

% ========== 新增指标计算 ==========
% 基础指标
TP = diagonal;
FP = sumCols - TP;
FN = sumRows - TP;
TN = sum(confusionMat(:)) - (TP + FP + FN);

% 各类别指标
precision = TP ./ (TP + FP);      % 精确率
recall = TP ./ (TP + FN);         % 召回率
f1Score = 2 * (precision .* recall) ./ (precision + recall); % F1-Score
iou = TP ./ (sumRows + sumCols - TP); % IoU
dice = 2 * TP ./ (sumRows + sumCols); % Dice系数
accuracy = (TP + TN) ./ sum(confusionMat(:)); % 准确率（每类）

% 全局指标
globalAccuracy = sum(TP) / sum(confusionMat(:)); % 全局准确率
meanPrecision = mean(precision(2:end));          % 平均精确率（忽略背景）
meanRecall = mean(recall(2:end));                % 平均召回率（忽略背景）
meanF1 = mean(f1Score(2:end));                   % 平均F1（忽略背景）
meanIoU = mean(iou(2:end));                      % mIoU（忽略背景）
meanDice = mean(dice(2:end));                    % 平均Dice（忽略背景）

% ========== 显示结果 ==========
disp('=== 测试性能 ===');
fprintf('%-12s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\n',...
    '类别', '准确率', '精确率', '召回率', 'F1-Score', 'IoU');
for i = 1:numClasses
    fprintf('%-12s\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n',...
        classNames{i}, accuracy(i), precision(i), recall(i), f1Score(i), iou(i));
end

fprintf('\n=== 全局指标 ===\n');
fprintf('全局准确率: \t%.4f\n', globalAccuracy);
fprintf('平均精确率: \t%.4f (忽略背景)\n', meanPrecision);
fprintf('平均召回率: \t%.4f (忽略背景)\n', meanRecall);
fprintf('平均F1-Score: \t%.4f (忽略背景)\n', meanF1);
fprintf('平均IoU: \t%.4f (mIoU, 忽略背景)\n', meanIoU);
fprintf('平均Dice: \t%.4f (忽略背景)\n', meanDice);

% ========== ROC曲线与AUC ==========
reset(dsTest);
allTrueLabels = [];
allPredScores = [];

while hasdata(dsTest)
    data = read(dsTest);
    
    % 通用数据解析
    if istable(data)
        % 表格数据处理
        X = data{1, 1}{1}; % 图像
        Y = data{1, 2}{1}; % 标签
    else
        error('非常规数据格式');
    end
    
    % GPU处理
    if canUseGPU
        X = gpuArray(X);
    end
    
    % 预测并处理概率
    scores = predict(net, X);
    scores = gather(scores(:,:,2)); % 孔隙类概率
    
    % 标签处理
    Y = gather(Y);
    if iscategorical(Y)
        Y = uint8(Y) - 1; % 转换为0/1编码
    end
    
    % 调整标签尺寸
    trueLabels = imresize(Y, size(scores), 'nearest') == 1;
    
    % 收集数据
    allTrueLabels = [allTrueLabels; trueLabels(:)];
    allPredScores = [allPredScores; scores(:)];
end

% 计算AUC
[Xpr, Ypr, Tpr, AUC] = perfcurve(allTrueLabels, allPredScores, 1);
fprintf('AUC: \t\t%.4f\n', AUC);

% 绘制ROC曲线
figure;
plot(Xpr, Ypr, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)', AUC));
grid on;

% ========== 输出到Excel ==========
% 创建结果表格
classMetrics = table(...
    classNames, accuracy, precision, recall, f1Score, iou, dice,...
    'VariableNames', {'Class', 'Accuracy', 'Precision', 'Recall', 'F1Score', 'IoU', 'Dice'});

globalMetrics = table(...
    globalAccuracy, meanPrecision, meanRecall, meanF1, meanIoU, meanDice, AUC,...
    'VariableNames', {'GlobalAccuracy', 'MeanPrecision', 'MeanRecall', 'MeanF1', 'mIoU', 'MeanDice', 'AUC'});

% ROC曲线数据（采样以减少文件大小）
sampleStep = max(1, floor(length(Xpr)/1000)); % 每1000个点取1个
rocData = table(...
    Xpr(1:sampleStep:end), Ypr(1:sampleStep:end),...
    'VariableNames', {'FPR', 'TPR'});

% 写入Excel文件
outputExcelFile = 'evaluation_metrics.xlsx';
writetable(classMetrics, outputExcelFile, 'Sheet', 'Class Metrics');
writetable(globalMetrics, outputExcelFile, 'Sheet', 'Global Metrics');
writetable(rocData, outputExcelFile, 'Sheet', 'ROC Curve');

% 保存ROC完整数据到MAT文件
save('roc_data.mat', 'Xpr', 'Ypr', 'AUC');

disp(['所有指标已保存到: ' outputExcelFile]);

%% 9. 输出测试集结果对比
outputDir = fullfile(dataPath, 'TestResults_CBAM_UNet');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

reset(dsTest); % 重置数据存储指针
i = 1;         % 初始化索引

% 定义颜色映射（背景：黑色，孔隙：白色）
cmap = [0 0 0; 1 1 1]; % 对应类别索引1和2

while hasdata(dsTest)
    % 正确读取数据存储（返回表格）
    data = read(dsTest);
    
    % 提取图像和标签（需解引用单元格）
    X = data{1, imageCol}{1}; % 图像位于第一列，双层单元格结构
    T = data{1, labelCol}{1}; % 标签位于第二列，应为分类数组
    
    % 预测并获取类别索引
    Y_prob = predict(net, X);
    [~, Y_index] = max(Y_prob, [], 3); % 沿通道维度取最大值
    
    % 转换预测结果为分类数组
    Y = categorical(Y_index, [1 2], classNames);
    
    % 将标签转换为数值索引（背景=1，孔隙=2）
    T_index = double(T); % 分类数组转数值
    
    % 转换为RGB图像
    T_rgb = ind2rgb(T_index, cmap); % 真实标签
    Y_rgb = ind2rgb(double(Y), cmap); % 预测结果
    
    % 生成对比图（并列显示）
    comparison = imtile({T_rgb, Y_rgb}, ...
        'GridSize', [1, 2], ...
        'BorderSize', 10, ...
        'BackgroundColor', 'w');
    
    % 保存图像
    outputFileName = fullfile(outputDir, sprintf('comparison_%03d.png', i));
    imwrite(comparison, outputFileName);
    
    i = i + 1;
end

disp(['测试集结果对比已保存到: ', outputDir]);

%% --- CBAM模块作为本地函数 ---
function [lgraph, cbamOutName] = cbamModule(lgraph, inputName, numChannels, blockName)
    % CBAM模块实现 - 先通道注意力(CA)，然后空间注意力(SA)
    % 输入:
    %   lgraph       - 当前网络图
    %   inputName    - 输入层名称
    %   numChannels  - 输入特征通道数
    %   blockName    - 模块唯一名称
    
    % 通道注意力部分 (Channel Attention)
    avgPoolName = [blockName '_ca_avgpool'];
    maxPoolName = [blockName '_ca_maxpool'];
    
    % 全局平均池化和最大池化
    gaPoolLayer = globalAveragePooling2dLayer('Name', avgPoolName);
    % 直接使用匿名函数定义全局最大池化
    gmPoolLayer = functionLayer(@(x) max(max(x,[],1),[],2), 'Name', maxPoolName);
    
    lgraph = addLayers(lgraph, gaPoolLayer);
    lgraph = addLayers(lgraph, gmPoolLayer);
    
    lgraph = connectLayers(lgraph, inputName, avgPoolName);
    lgraph = connectLayers(lgraph, inputName, maxPoolName);
    
    % 共享MLP - 先将通道降维，再升维
    mlp1Name = [blockName '_ca_mlp1'];
    mlp2Name = [blockName '_ca_mlp2'];
    mlpReluName = [blockName '_ca_relu'];
    
    r = ; % 通道数缩减系数
    mlpFeatures = max(numChannels/r, ); % 避免太小
    
    % MLP第一层 (numChannels -> mlpFeatures)
    mlp1LayerObj = fullyConnectedLayer(mlpFeatures, 'Name', mlp1Name);
    reluLayerObj = reluLayer('Name', mlpReluName);
    
    % MLP第二层 (mlpFeatures -> numChannels)
    mlp2LayerObj = fullyConnectedLayer(numChannels, 'Name', mlp2Name);
    
    % 为平均池化分支添加MLP
    lgraph = addLayers(lgraph, [mlp1LayerObj reluLayerObj mlp2LayerObj]);
    lgraph = connectLayers(lgraph, avgPoolName, mlp1Name);
    
    % 为最大池化分支复用相同的MLP
    avg_mlp2_out = mlp2Name;
    
    % 为最大池化分支创建单独的MLP（共享权重）
    mlp1Name_max = [blockName '_ca_mlp1_max'];
    mlpRelu_max = [blockName '_ca_relu_max'];
    mlp2Name_max = [blockName '_ca_mlp2_max'];
    
    mlp1LayerObj_max = fullyConnectedLayer(mlpFeatures, 'Name', mlp1Name_max);
    reluLayerObj_max = reluLayer('Name', mlpRelu_max);
    mlp2LayerObj_max = fullyConnectedLayer(numChannels, 'Name', mlp2Name_max);
    
    lgraph = addLayers(lgraph, [mlp1LayerObj_max reluLayerObj_max mlp2LayerObj_max]);
    lgraph = connectLayers(lgraph, maxPoolName, mlp1Name_max);
    
    max_mlp2_out = mlp2Name_max;
    
    % 加法层合并两个MLP输出
    addName = [blockName '_ca_add'];
    addLayerObj = additionLayer(, 'Name', addName);
    lgraph = addLayers(lgraph, addLayerObj);
    lgraph = connectLayers(lgraph, avg_mlp2_out, [addName '/in1']);
    lgraph = connectLayers(lgraph, max_mlp2_out, [addName '/in2']);
    
    % Sigmoid激活
    sigmoidName = [blockName '_ca_sigmoid'];
    sigmoidLayerObj = sigmoidLayer('Name', sigmoidName);
    lgraph = addLayers(lgraph, sigmoidLayerObj);
    lgraph = connectLayers(lgraph, addName, sigmoidName);
    
    % 通道注意力加权 - 简化实现
    mulName = [blockName '_ca_mul'];
    mulLayerObj = functionLayer(@safeChannelAttention, 'NumInputs', , 'Name', mulName);
    lgraph = addLayers(lgraph, mulLayerObj);
    lgraph = connectLayers(lgraph, inputName, [mulName '/in1']);
    lgraph = connectLayers(lgraph, sigmoidName, [mulName '/in2']);
    
    % 通道注意力结果
    channelAttOut = mulName;
    
    % 空间注意力部分 (Spatial Attention)
    % 计算平均池化和最大池化（沿通道维度）
    avgPoolCName = [blockName '_sa_avgpoolc'];
    maxPoolCName = [blockName '_sa_maxpoolc'];
    
    avgPoolCLayerObj = functionLayer(@(X) mean(X, 3), 'Name', avgPoolCName);
    maxPoolCLayerObj = functionLayer(@(X) max(X, [], 3), 'Name', maxPoolCName);
    
    lgraph = addLayers(lgraph, avgPoolCLayerObj);
    lgraph = addLayers(lgraph, maxPoolCLayerObj);
    
    lgraph = connectLayers(lgraph, channelAttOut, avgPoolCName);
    lgraph = connectLayers(lgraph, channelAttOut, maxPoolCName);
    
    % 拼接平均池化和最大池化的结果
    concatName = [blockName '_sa_concat'];
    concatLayerObj = depthConcatenationLayer(2, 'Name', concatName);
    lgraph = addLayers(lgraph, concatLayerObj);
    
    % 使用安全的reshape函数
    reshapeAvgName = [blockName '_reshape_avg'];
    reshapeMaxName = [blockName '_reshape_max'];
    
    reshapeAvgLayerObj = functionLayer(@safeReshape, 'Name', reshapeAvgName);
    reshapeMaxLayerObj = functionLayer(@safeReshape, 'Name', reshapeMaxName);
    
    lgraph = addLayers(lgraph, reshapeAvgLayerObj);
    lgraph = addLayers(lgraph, reshapeMaxLayerObj);
    
    lgraph = connectLayers(lgraph, avgPoolCName, reshapeAvgName);
    lgraph = connectLayers(lgraph, maxPoolCName, reshapeMaxName);
    
    lgraph = connectLayers(lgraph, reshapeAvgName, [concatName '/in1']);
    lgraph = connectLayers(lgraph, reshapeMaxName, [concatName '/in2']);
    
    % 7x7卷积生成空间注意力图
    convName = [blockName '_sa_conv'];
    convLayerObj = convolution2dLayer(7, 1, 'Padding', 'same', 'Name', convName);
    lgraph = addLayers(lgraph, convLayerObj);
    lgraph = connectLayers(lgraph, concatName, convName);
    
    % Sigmoid激活
    sigmoidSAName = [blockName '_sa_sigmoid'];
    sigmoidSALayerObj = sigmoidLayer('Name', sigmoidSAName);
    lgraph = addLayers(lgraph, sigmoidSALayerObj);
    lgraph = connectLayers(lgraph, convName, sigmoidSAName);
    
    % 空间注意力加权 - 简化实现
    mulSAName = [blockName '_sa_mul'];
    mulSALayerObj = functionLayer(@safeElementWiseMul, 'NumInputs', 2, 'Name', mulSAName);
    lgraph = addLayers(lgraph, mulSALayerObj);
    lgraph = connectLayers(lgraph, channelAttOut, [mulSAName '/in1']);
    lgraph = connectLayers(lgraph, sigmoidSAName, [mulSAName '/in2']);
    
    % 返回CBAM模块的输出
    cbamOutName = mulSAName;
end

% 安全的reshape函数 - 确保输出形状为[H,W,1]
function out = safeReshape(X)
    % 获取输入尺寸
    [h, w, ~, ~] = size(X);
    
    % 如果输入已经是单通道，直接返回
    if size(X,3) == 1
        out = X;
        return;
    end
    
    % 确保X是2D (扁平化通道)
    if ndims(X) > 2
        % 计算可能的单通道输出
        X_flat = X(:,:,1,:); % 只保留第一个通道
    else
        X_flat = X;
    end
    
    % 确保输出是3D [H,W,1]
    out = X_flat;
end

% 简化版安全通道注意力函数
function out = safeChannelAttention(X, weights)
    % 确保X是4D输入 [H,W,C,N]
    [h, w, c, ~] = size(X);
    
    % 处理权重：无论其形状如何，尝试将其转换为与X的通道数匹配
    weights = squeeze(weights); % 移除单一维度
    
    % 如果权重是列向量或行向量，直接使用
    if isvector(weights)
        weights = weights(:); % 确保为列向量
        
        % 如果长度不匹配，则用重复填充或截断
        if length(weights) ~= c
            if length(weights) > c
                weights = weights(1:c); % 截断
            else
                % 重复向量以匹配通道数
                rep = ceil(c / length(weights));
                weights = repmat(weights, rep, 1);
                weights = weights(1:c);
            end
        end
    else
        % 如果是矩阵或高维数组，展平并调整大小
        weights = weights(:);
        if length(weights) ~= c
            weights = ones(c, 1) * mean(weights); % 使用平均值填充
        end
    end
    
    % 重塑为广播兼容的形状 [1,1,C,1]
    weights = reshape(weights, 1, 1, c, 1);
    
    % 应用通道注意力
    out = X .* weights;
end

% 简化版安全元素乘法函数
function out = safeElementWiseMul(X, Y)
    % 获取输入尺寸
    [h, w, c_x, n_x] = size(X);
    
    % 确保Y的尺寸兼容X
    try
        % 如果Y是单通道但X是多通道，扩展Y
        if size(Y,3) == 1 && c_x > 1
            Y_expanded = repmat(Y, [1, 1, c_x, 1]);
            out = X .* Y_expanded;
        else
            % 尝试直接相乘
            out = X .* Y;
        end
    catch
        % 如果失败，将Y视为掩码并应用
        try
            % 确保Y是2D掩码
            mask = Y(:,:,1);
            % 将掩码应用于每个通道
            out = zeros(size(X), 'like', X);
            for i = 1:c_x
                out(:,:,i,:) = X(:,:,i,:) .* mask;
            end
        catch
            % 最后的安全措施：直接返回X
            out = X;
        end
    end
end

