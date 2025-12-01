for dir in */; do
	tar --exclude="${dir}info.log" -czvf "${dir%/}.tar.gz" "$dir"
done
