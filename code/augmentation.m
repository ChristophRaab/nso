function [Xs,Ys] = augmentation(Xs,Xt,Ys)
%BASIS_TRANSFER Summary of this function goes here
 C = unique(Ys,'stable');
    sizeC = size(C,1);
    if size(Xs,1) < size(Xt,1)
        fprintf("Smaller")
        data = [];
        label = [];
        diff = size(Xt,1) - size(Xs,1);
        sampleSize = floor(diff / sizeC);
        for c = C'
            idxs = find(Ys == c);
            classData= Xs(idxs,:);
            m = mean(classData); sd = std(classData);
            augmentationData = mvnrnd(m,sd,sampleSize);
            data = [data; classData;augmentationData];
            label = [label;ones(size(classData,1),1)*c;ones(sampleSize,1)*c];
        end
        sampleSize = mod(diff,sizeC);
        c = C(end);
        idxs = find(Ys == c);
        classData= Xs(idxs,:);
        m = mean(classData); sd = std(classData);
        augmentationData = mvnrnd(m,sd,sampleSize);
        data = [data;augmentationData];
        label = [label;ones(size(augmentationData,1),1)*c];
        Xs = data;Ys = label;
    end
   
    if size(Xs,1) > size(Xt,1)
        fprintf("Not Smaller\n");
        data = [];
        label = [];
        diff = size(Xs,1) - size(Xt,1);
        sampleSize = floor( size(Xt,1) / sizeC);
        for c = C'
            idxs = find(Ys == c); 
            classData= Xs(idxs,:);
            if size(idxs,1) > sampleSize
                y = randsample(size(classData,1),sampleSize);
                classData = classData(y,:);
            end
            data = [data; classData];
            label = [label;ones(size(classData,1),1)*c];
        end
        sampleSize = abs(size(data,1)-size(Xt,1));
        c = C(end);
        idxs = find(Ys == c);
        classData= Xs(idxs,:);
        m = mean(classData); sd = std(classData);
        augmentationData = mvnrnd(m,sd,sampleSize);
        data = [data;augmentationData];
        label = [label;ones(size(augmentationData,1),1)*c];
        Xs = data;Ys = label;
    end
end

