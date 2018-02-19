# This file only compiles on Phong's mac where the Eigen library is located at ~/libs/eigen
# Set your include path appropriately on your machine

netcdf_home=/usr/local/Cellar/netcdf/4.4.1.1_5/

mpic++ -o main main.cpp -I ~/libs/eigen -I ${netcdf_home}/include -L ${netcdf_home}/lib -lnetcdf