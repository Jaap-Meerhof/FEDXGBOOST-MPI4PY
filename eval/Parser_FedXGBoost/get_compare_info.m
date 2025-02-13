function out = get_compare_info(handle)
    nTest = numel(handle);
    nFeature = zeros(nTest, 1);
    dt = zeros(nTest, 1);
    nUsers = zeros(nTest, 1);
    nBytes = zeros(nTest, 1);
   
    for i = 1: nTest
        nFeature(i) = handle{i}.nFeature;
        dt(i) = handle{i}.dtTree;
        nUsers(i) = handle{i}.nUsers;
        nBytes(i) = handle{i}.nBytes;
        lambda(i) = handle{i}.param.xgbParam.lambda;
        gamma(i) = handle{i}.param.xgbParam.gamma;
        
    end

    out.nFeature = nFeature;
    out.dt = dt;
    out.nUsers = nUsers;
    out.nBytes = nBytes;
    
    try
        out.lambda = lambda;
        out.gamma = gamma;
    catch
        
    end
end

