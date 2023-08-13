// alert("hello")


$('#left-box').click(function(){
	const allBtns = $('.toToggle');
    allBtns.click(function(){
        if ($(".right-box").hasClass("makeInivisible")){
            $(".right-box").addClass("addedClass");
        }
        
    })
})  

const allBtns = $('.btn-chart');
for (const btn of allBtns){
    console.log($(btn).val());
    
    if ($(btn).click(function(){
        
        for (const img of $("img")){
            $(img).removeClass("image-visible");
            $(img).addClass("graph-image");
        }


        if ($(btn).attr("id") == "SAvT"){

            $("#Surface-Area").removeClass("graph-image");
            $("#Surface-Area").addClass("image-visible");
        }else if ($(btn).attr("id") == "VvT"){
            $("#Volume").removeClass("graph-image");
            $("#Volume").addClass("image-visible");
        }
    }));
    
}



