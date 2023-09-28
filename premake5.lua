workspace "TurboSpork"
    configurations { "debug", "release" }
    startproject "TurboSpork"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "TurboSpork"
    language "C"
    location "src"
    kind "ConsoleApp"

    includedirs {
        "src",
        "src/third_party"
    }

    files {
        "src/**.h",
        "src/**.c",
    }

    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    targetdir ("bin/" .. outputdir)

    warnings "Extra"
    architecture "x64"
    toolset "clang"

    filter "system:linux"
        links {
            "m", "X11", "GL", "GLX"
        }

    filter { "system:windows", "action:*gmake*", "configurations:debug" }
        linkoptions { "-g" }

    filter "configurations:debug"
        symbols "On"

        defines {
            "DEBUG"
        }

    filter "configurations:release"
        optimize "On"
        defines { "NDEBUG" }

    filter "system:windows"
        systemversion "latest"

        links {
            "gdi32", "user32", "opengl32"
        }