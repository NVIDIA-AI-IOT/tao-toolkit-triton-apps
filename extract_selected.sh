mkdir ./selected
mkdir ./selected/JPEGImages
mkdir ./selected/Annotations
cp -r ./VOC2012/ImageSets/ ./selected/

while IFS= read -r line; 
do 
    mv ./VOC2012/JPEGImages/$line.jpg ./selected/JPEGImages/$line.jpg
    mv ./VOC2012/Annotations/$line.xml ./selected/Annotations/$line.xml
done < ./VOC2012/ImageSets/Main/test.txt
