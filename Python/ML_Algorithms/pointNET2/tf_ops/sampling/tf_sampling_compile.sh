nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/snaffe/anaconda3/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/include/ -I /usr/local/cuda/include -I /home/snaffe/anaconda3/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L/home/snaffe/anaconda3/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/ -l:libtensorflow_framework.so -O2


g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
-I /opt/conda/lib/python3.7/site-packages/tensorflow/include/ -I /usr/local/cuda-10.0/include \
-I /opt/conda/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart \
-L /usr/local/cuda-10.0/lib64/ -L/opt/conda/lib/python3.7/site-packages/tensorflow/ -l:libtensorflow_framework.so.1 -O2
