#/bin/sh

wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
rm -rf leaves.zip

chmod 755 -R images

mkdir -p images/apple
mv images/Apple_Black_rot images/apple/black_rot
mv images/Apple_healthy images/apple/healthy
mv images/Apple_rust images/apple/rust
mv images/Apple_scab images/apple/scab

mkdir -p images/grape
mv images/Grape_Black_rot images/grape/black_rot
mv images/Grape_Esca images/grape/esca
mv images/Grape_healthy images/grape/healthy
mv images/Grape_spot images/grape/spot
