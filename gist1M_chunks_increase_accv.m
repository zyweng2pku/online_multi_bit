% test for gist1M
% test when more and more streaming data added to the database
clc;
clear all;
% run('E:\wzy\博士阶段\projects\toolbox\vlfeat-0.9.19-bin\vlfeat-0.9.19\toolbox\vl_setup')
% extract data
% addpath('mex');
display('data extraction');


% % extract ann1mgist
addpath('..\dataset\anngist1m');
addpath('..\common_code\util_read');
X_data = fvecs_read('gist_base.fvecs');
X_data = X_data'; % N * D
X_test = fvecs_read('gist_query.fvecs');
X_test = X_test'; % N * D
gt = ivecs_read('gist_groundtruth.ivecs');
gt = gt';
load gt;

% prepare data
% display('data preparation');


% WtrueTestTraining = bsxfun(@plus, gt(:, 1:NumberNeighbors), 0);%for our gt
% WtrueTestTraining = bsxfun(@plus, gt, 1);%for original gt from sift1M and
% GIST1M

% prepare data for whrank
% [ X_train, WtrueWhrankTraining ] = dataPreWhrank( X_data, NumberNeighbors );
% save demo3_training_whrank X_train WtrueWhrankTraining
% clear X_train WtrueWhrankTraining;
% load demo3_training_whrank
% load groundTruth

 NumberNeighbors = 1000; % ground_truth
%% load data
% load train_data
% groundTruth = bsxfun(@plus, gt(:, 1:NumberNeighbors), 0);%for our gt
Xtraining =  X_data;
Xtest = X_test;
load train_data

load groundTruth;
groundTruth = int32(groundTruth(:,1: NumberNeighbors));
%% load normalized data

% load nor_train_data
% Xtest = normr(X_test);
% Xtraining = normr(X_data);

% l = randperm(size(Xtraining, 1));
% X = Xtraining(l(1:1000), :);
% thresdist = 0;
% for i = 1 : size(X, 1)
%     s = bsxfun(@minus, X(i, :), Xtraining);
%     s = sum(s.^2, 2);
%     s = sqrt(s);
%     [s, ind] = sort(s);
%     thresdist = thresdist + s(NumberNeighbors);
% end
% thresdist = thresdist / 1000;
% 
% seq = zeros(size(Xtraining, 1), 3);
% for i = 1 : size(Xtraining, 1)
%     seq(i, 1) = unidrnd(size(Xtraining, 1));
%     seq(i, 2) = unidrnd(size(Xtraining, 1));
%     s = Xtraining(seq(i, 1), :) - Xtraining(seq(i, 2), :);
%     s = sqrt(sum(s.^ 2));
%     if s <= thresdist
%         seq(i, 3) = 1;
%     end
%         seq(i, 3) = -1;
%     
% end
% save('seq_20000', 'thresdist', 'seq');
%% 
% m = mean(Xtraining);
% Xtraining = bsxfun(@minus, Xtraining, m);
% Xtest = bsxfun(@minus, Xtest, m);
rmpath('..\common_code\util_read');
rmpath('..\dataset\anngist1m');





clear gt;
clear X_test;
clear X_data;

% gt = zeros( 1000, 1000 );
% for i = 1:1000
%     dis = bsxfun(@minus, Xtest(i,:), Xtraining);
%     dis = dis.^2;
%     dis = sum(dis,2);
%     [~,Index] = sort(dis');
%     gt(i,:) = Index(1:1000);
% end

% test
%display('test');

NNs = [1000];
% groundTruth = zeros(size(Xtest, 1), NumberNeighbors);
% for i = 1 : size(Xtest, 1)
% %     l = find(label(size(Xtraining, 1)+i) ==label(1:size(Xtraining, 1)));
%     groundTruth(i, :) = WtrueTestTraining(i, :);
% end
seg = 1000;
groundTruth = int32(groundTruth(:,1: NumberNeighbors));
printhot = zeros(seg, 1);
i = 1;
% while i <= seg
%     printhot(i) = 1;
%     i = i * 2;
% end
printhot(1:1:10) = 1;
printhot(10:10:100) = 1;
 printhot(seg) = 1;
% bits = [64];
bits = [64];
recalls = [];
mAPs = [];
mAPs1 = [];
presicions = [];
eps = [];
training_time = [];
pres = [];
pres1 = [];
% ll = [0.01, 0.1, 1, 10, 100];
% for parai = 1 : length(ll)
% methods = {'AdaptH','MIH+TU'};
methods = { 'OMQH'};
for methodi = 1 : length(methods)
for ir = 1 : 1
    for ii = 1 : length(bits)
        bit = bits(ii);
        method = methods{methodi};
        switch(method)
        %% OSH
            case 'OSH'
            addpath('D:\project\hash\OnlineHashingComparedMethod\OnlineHashingComparedMethod\OSH');
            tic;
            l = 300;
            rounds = seg;
            model.Q = zeros(size(Xtraining, 2), l);  % osh中数据是D*N
            model.niu = 0;
            model.acc_num = 0;
            for i  = 1 : 100
                
                
                len = size(Xtraining, 1) / rounds;
                st = (i - 1) * len + 1;
                ed = i * len;
                model = OSH(Xtraining(int32(st):int32(ed),:)', model.Q, model.niu, model.acc_num);
                
               
                if printhot(i) == 1
                                Tdata = toc;
                                training_time = [training_time, Tdata];
                                Q = model.Q';

                                [pc, l] = eigs(double(cov(double(Q))),bit);

                                R =  randn(bit,bit);
                                [R,~,~] = svd(R);

                                g = zeros(size(Xtest, 1), 1000);
                                for f = 1 : size(Xtest, 1)
                                    ss = bsxfun(@minus, Xtraining(1:ed, :), Xtest(f,:));
                                    ss = ss .^ 2;
                                    ss = sum(ss, 2);
                                    [~, ind] = sort(ss);
                                    g(f, :) = ind(1:1000);
                                end
                                g = int32(g);
                                for ni = 1 : length(NNs)
                                    NN = NNs(ni);
                                    Y1 = zeros(ed, bit);
                                    Y2 = zeros(size(Xtest, 1), bit);
                                    mm = mean(Xtraining(1:ed, :), 1);
                                    X1 = (bsxfun(@minus, Xtraining(1:ed, :), mm)) * pc * R;
                                    X2 = bsxfun(@minus, Xtest, mm) * pc * R;
                                    Y1(X1 >= 0) = 1;
                                    Y2(X2 >= 0) = 1;
                                    Bdata = compactbit(Y1);
                                    Btest = compactbit(Y2);      
                                    mAP = cal_chunk_map(Bdata, Btest, g);
                                    mAPs = [mAPs mAP];
%                                     mAP1 = cal_chunk_map(Bdata, Btest, groundTruth(:,1:1));
%                                     mAPs1 = [mAPs1 mAP1];
                                    pre = cal_chunk_pre(Bdata, Btest, g, 100);
                                    pres = [pres pre]
%                                     pre1 = cal_chunk_pre(Bdata, Btest, groundTruth(:,1:1), 1);
%                                     pres1 = [pres1 pre1]
                                end
                                tic;
                end
                
            end
 
            rmpath('D:\project\hash\OnlineHashingComparedMethod\OnlineHashingComparedMethod\OSH');
        
        %% online multi-bit quantization hashing
            case 'OMQH'
                
            addpath('D:\project\hash\OnlineHashingComparedMethod\OnlineHashingComparedMethod\OSH');
            
            l = 300;
            rounds = seg;
            model.Q = zeros(size(Xtraining, 2), l);  % osh中数据是D*N
            model.niu = 0;
            model.acc_num = 0;
            tic;
            for i  = 1 : 10
                
                
                len = size(Xtraining, 1) / rounds;
                st = (i - 1) * len + 1;
                ed = i * len;
                model = OSH(Xtraining(int32(st):int32(ed),:)', model.Q, model.niu, model.acc_num);
                 if printhot(i) == 1
                         Q = model.Q';
                        cur_N = i * len;
                        mm = mean(Xtraining(1:cur_N,:), 1);
                        [pc, eig_values] = eigs(double((double(Q' * Q)) ./ cur_N), bit);
                        b = bit;
                        
                        count = 0;
                        eig_values = zeros(b, 1);
                        s = bsxfun(@minus, Xtraining(1:cur_N,:), mm) * pc;
                        for l = 1 : bit
                            ss = bsxfun(@minus, s(:, l), mm(l));
                            ss = ss .^ 2;
                            ss = sum(ss) ;
                            ss = ss ./ cur_N;
%                             eig_values(l) = sqrt(ss);
                            eig_values(l) = ss;
                            if eig_values(l) < 1e-6
                                eig_values(l) = 1e-6;
                            end                            
                        end
                        thres = 0.7;
                        v = sum(eig_values);
                        vc = cumsum(eig_values);
                        vc = vc ./ v;
                        for l = length(vc) : -1 : 1 
                            if vc(l) <= thres
                                d = l;
                                break;
                            end
                        end
                        b_dist = ones(d, 1);
%                         e = (eig_values(1:d)) ./ 2;
                        e = sqrt(eig_values(1:d)) ./ 2;
                        for l = 1 : bit - d
                            [~, ind] = max(e);
                            e(ind) = e(ind) / 2;
                            b_dist(ind) = b_dist(ind) + 1;
                        end            
%                         b_dist = [3,2,2,2,1,1,1,1,1,1,1];
                        dims = cell(length(b_dist), 1);

                        for l = 1 : length(b_dist)
                            
                            dims{l}.bit = b_dist(l);
                            dims{l}.parts = [];
                            pd = makedist('Normal', 'mu', 0, 'sigma', sqrt(eig_values(l)));
                            for ll = 1 : 2^dims{l}.bit - 1
                              dims{l}.parts = [dims{l}.parts, ll/(2^dims{l}.bit)]; 
                            end
                            dims{l}.parts = icdf(pd, dims{l}.parts);
                            dims{l}.part_values = [];
                            for ll = 1 : 2^dims{l}.bit
                                dims{l}.part_values = [dims{l}.part_values, ((ll - 1) * 2 + 1) / (2^(dims{l}.bit+1))];
                            end
                            dims{l}.part_values = icdf(pd, dims{l}.part_values);
                        end
                        Tdata = toc;
                        Btest = Xtest * pc(:, 1:length(b_dist));
                        Btraining = zeros(size(Xtraining, 1), length(b_dist));
                        Dtraining = Xtraining * pc(:, 1:length(b_dist));
                        training_time = [training_time, Tdata];
                        for l = 1 : length(b_dist)
                            t = bsxfun(@minus, Dtraining(:, l), dims{l}.parts);
                            Btraining(:, l) = dims{l}.part_values(2^dims{l}.bit);
                            for j = 2^dims{l}.bit-1 : -1 : 1
                                ind = find(t(:, j) <= 0);
                                Btraining(ind, l) = dims{l}.part_values(j);
                            end
                        end
                        g = zeros(size(Xtest, 1), 1000);
                        for f = 1 : size(Xtest, 1)
                            ss = bsxfun(@minus, Xtraining(1:ed, :), Xtest(f,:));
                            ss = ss .^ 2;
                            ss = sum(ss, 2);
                            [~, ind] = sort(ss);
                            g(f, :) = ind(1:1000);
                        end
                        g = int32(g);                        
                        
                        for ni = 1 : length(NNs)
                            NN = NNs(ni);
                            [mAP, pre] = calculate_map_multi(Btraining(1:ed,:), Btest, g);
                            mAPs = [mAPs, mAP]
                            pres = [pres, pre]; 
                        end
                        tic;
                 end
        
                
            end
            


            rmpath('D:\project\hash\OnlineHashingComparedMethod\OnlineHashingComparedMethod\OSH');

  
        end

         %% evaluation mAP
%             addpath('../mex');
% 
%             NearestNum = size(Bdata, 1);
%     %         NearestNum = 100;
% 
%             nSlice = 1;
%             jSlice = size(Btest,1) / nSlice;
% 
%             B = Btest;
%             p = 0;
%             p1 = 0;
%             ep = 0;
%             mAP = 0;
% 
% 
% 
%             for iSlice = 1 : nSlice
%                 stSlice = (iSlice - 1) * jSlice + 1;
%                 edSlice = iSlice * jSlice; 
% 
%                 B = Btest(stSlice:edSlice, :);
% 
%                 g = groundTruth(stSlice:edSlice, :);
% 
%                 AP =mexBinarySortMAP(Bdata', B', g', int32(NearestNum));
%                 mAP = mAP + AP;
% 
% 
%             end
% 
%             mAP =  mAP / nSlice;
%             mAPs = [mAPs mAP]
%             training_time = [training_time, Tdata];
% 
% 
%             rmpath('../mex');
    end
end
end


