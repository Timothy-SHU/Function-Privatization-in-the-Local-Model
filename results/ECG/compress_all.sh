for file in */; do
	tar -czvf "${file%/}.tar.gz" "$file"
done
