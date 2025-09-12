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

%% 4. 定义U-Net模型
numClasses = ;
lgraph = unetLayers(inputSize, numClasses,...
    'EncoderDepth', ,...
    'NumFirstEncoderFilters', );

%% 5. 训练配置（关键修正）
validationFrequency = ; % 显式定义验证频率变量

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
save('unet_soil_model.mat', 'net');

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
    'OutputSize', inputSize(),...
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
outputDir = fullfile(dataPath, 'TestResults');
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
    X = data{1, 1}{1}; % 图像位于第一列，双层单元格结构
    T = data{1, 2}{1}; % 标签位于第二列，应为分类数组
    
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
