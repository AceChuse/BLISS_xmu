    #include<cuda_runtime.h>    
    #include<iostream>  
    using namespace std;  
    const int nMax = 30000;  
    __global__ void addKernel(float *aaa,float *bbb, float *ccc)  
    {  
        //int i = blockIdx.x;  
        int i = threadIdx.x + blockIdx.x*blockDim.x;  
        ccc[i] = 0;  
        if (i < nMax)for (int j = 0; j < 500; j++)ccc[i] += aaa[i] * bbb[i];  
    }  
    void add(float *a, float *b,float *c,int i){  
        for (int j = 0; j<500; j++) c[i] += a[i] * b[i];  
    }  
    int main(){  
        float a[nMax], b[nMax], c[nMax];  
        float *devA, *devB, *devC;  
        clock_t startT, endT;  
        for (int i = 0; i < nMax; i++){  
            a[i] = i*1.010923;  
            b[i] = 2.13*i;  
        }  
        startT = clock();  
        cudaMalloc((void**)&devA, nMax*sizeof(float));  
        cudaMalloc((void**)&devB, nMax*sizeof(float));  
        cudaMalloc((void**)&devC, nMax*sizeof(float));  
        endT = clock();  
        cout << "分配设备空间耗时 " << endT - startT << "ms"<<endl;  
      
      
        startT = clock();  
        cudaMemcpy(devA, a,nMax*sizeof(float),cudaMemcpyHostToDevice);  
        cudaMemcpy(devB, b, nMax*sizeof(float), cudaMemcpyHostToDevice);  
        endT = clock();  
        cout << "数据从主机写入设备耗时 " << endT - startT << "ms" << endl;  
      
        startT = clock();  
      
        cudaEvent_t start1;  
        cudaEventCreate(&start1);  
        cudaEvent_t stop1;  
        cudaEventCreate(&stop1);  
        cudaEventRecord(start1, NULL);  
      
        addKernel<<<60,501>>>(devA, devB, devC);  
      
        cudaEventRecord(stop1, NULL);  
        cudaEventSynchronize(stop1);  
        float msecTotal1 = 0.0f;  
        cudaEventElapsedTime(&msecTotal1, start1, stop1);  
        //cout << msecTotal1 << "ddd" << endl;  
        endT = clock();  
        cout << "GPU计算耗时 " << msecTotal1 << "ms" << endl;  
      
        startT = clock();  
        cudaMemcpy(c, devC, nMax*sizeof(float), cudaMemcpyDeviceToHost);  
        endT = clock();  
        cout << "数据从设备写入主机耗时 " << endT - startT << "ms" << endl;  
      
        cout <<"GPU计算结果 "<< c[nMax - 1] << endl;  
        for (int i = 0; i < nMax; i++){  
            a[i] = i*1.010923;  
            b[i] = 2.13*i;  
            c[i] = 0;  
        }  
        startT = clock();  
        for (int i = 0; i < nMax; i++){  
            add(a, b, c, i);  
        }  
        endT = clock();  
        cout << "CPU计算耗时 " << endT - startT << "ms" << endl;  
        cout << "CPU计算结果 " << c[nMax - 1] << endl;  
      
            //释放在设备上分配的空间  
        cudaFree(devA);  
        cudaFree(devB);  
        cudaFree(devC);  
        cin >> a[0];  
        return 0;  
    }  
