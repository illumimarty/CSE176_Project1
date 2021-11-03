% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Create a version of MNIST with smaller images. This makes the algorithms
% run faster while achieving comparable results, and it makes the covariance
% matrix of the data full rank, which is convenient for some algorithms.

load MNIST.mat; [N,D] = size(train_fea);

% Reduce dimension by subsampling & cropping
II = reshape(1:D,28,28); JJ = II(1:2:end,1:2:end);	% 28x28 -> 14x14
JJ = JJ(3:end-2,3:end-2); JJ = JJ(:);			% remove 2-pixel margin
train_fea1 = train_fea(:,JJ); train_gnd1 = train_gnd;
test_fea1 = test_fea(:,JJ); test_gnd1 = test_gnd;

save MNISTmini.mat train_fea1 train_gnd1 test_fea1 test_gnd1

X = double(train_fea)/255; X1 = double(train_fea1)/255;
size(X,2), rank(cov(X))					% original
size(X1,2), rank(cov(X1))				% after reduction

DD = sqrt(size(X,2)); DD1 = sqrt(size(X1,2));
% Plot some pairs of (original,reduced) images
figure(1); clf; colormap(gray(256));
for k=randperm(size(X,1),10)
  subplot(1,2,1); imagesc(reshape(X(k,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image; title(['original #' num2str(k)]);
  subplot(1,2,2); imagesc(reshape(X1(k,:),DD1,DD1),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image; title(['reduced #' num2str(k)]);
  pause
end

