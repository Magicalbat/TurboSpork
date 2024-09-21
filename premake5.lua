workspace "TurboSpork"
    architecture "x64"
    configurations { "debug", "release" }
    startproject "TurboSpork"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "TurboSpork"
    language "C"
    location "turbospork"
    kind "StaticLib"

    includedirs {
        "turbospork/include",
        "turbospork/third_party"
    }

    files {
        "turbospork/**.h",
        "turbospork/**.c",
    }

    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    targetdir ("bin/" .. outputdir)

    warnings "Extra"
    architecture "x64"
    toolset "clang"

    filter "system:linux"
        pic "On"
        links {
            "m", "pthread"
        }

    filter "system:windows"
        systemversion "latest"

        links {
            "gdi32", "user32", "Bcrypt"
        }

    -- Workaround for clang on windows
    filter { "system:windows", "action:*gmake*", "configurations:debug" }
        linkoptions { "-g" }
        
    filter { "action:not vs*", "configurations:debug" }
    	buildoptions { "-fsanitize=address" }
    	linkoptions { "-fsanitize=address" }

    filter "configurations:debug"
        symbols "On"
        defines {
            "DEBUG"
        }

    filter "configurations:release"
        optimize "On"

        defines { "NDEBUG" }

project "Examples"
    language "C"
    location "examples"
    kind "ConsoleApp"

    includedirs {
        "examples",
        "examples/third_party",
        "turbospork/include",
    }

    files {
        "examples/**.h",
        "examples/**.c",
    }

    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    targetdir ("bin/" .. outputdir)

    warnings "Extra"
    architecture "x64"
    toolset "clang"

    links {
        "turbospork"
    }

    filter "system:linux"
        links {
            "m",
        }

    filter "system:windows"
        systemversion "latest"

        links {
            "gdi32", "user32", "Bcrypt"
        }

    -- Workaround for clang on windows
    filter { "system:windows", "action:*gmake*", "configurations:debug" }
        linkoptions { "-g" }
        
    filter { "action:not vs*", "configurations:debug" }
        buildoptions { "-fsanitize=address" }
    	linkoptions { "-fsanitize=address" }

    filter "configurations:debug"
        symbols "On"
        defines {
            "DEBUG"
        }

    filter "configurations:release"
        optimize "On"

        defines { "NDEBUG" }

