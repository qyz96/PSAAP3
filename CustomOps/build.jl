function install_had()
    change_directory()
    git_repository("https://github.com/kailaix/had", "had")
end


install_had()
change_directory(joinpath(@__DIR__, "build"))
require_file("build.ninja") do 
    ADCME.cmake()
end
require_library("psaap3") do 
    ADCME.make()
end