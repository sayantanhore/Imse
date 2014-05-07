function pathGenerator(url){
	
	var PATH;
	
	if(url.search("/imse_dev2/") != -1){
		PATH = "/imse_dev2/"
	}
	else if(url.search("/imse_test/") != -1){
		PATH = "/imse_test/"
	}
	else if(url.search("/imse_dev/") != -1){
		PATH = "/imse_dev/"
	}
	
	return PATH;
}
