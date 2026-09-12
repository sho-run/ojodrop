# Third-Party Notices & Attribution — OjoDrop

## BeatDrop extended waveform geometry

Base waveform modes 8–17 adapt geometry from
[BeatDrop](https://github.com/OfficialIncubo/BeatDrop-Music-Visualizer), commit
`945ae10ecf928d24717b64f4e1a69b2c100c4829`.
Copyright (c) 2018 Maxim Volskiy and individual contributors. The complete
BSD-3-Clause notice is retained in
[`licenses/BeatDrop-BSD-3-Clause.txt`](./licenses/BeatDrop-BSD-3-Clause.txt) and
[`extended_waveforms.rs`](./crates/particle-milkdrop/src/extended_waveforms.rs).
The notice must accompany binary distributions. No endorsement is implied.

OjoDrop is MIT-licensed (see [`LICENSE`](./LICENSE)). It stands on the shoulders
of the people who built MilkDrop, Butterchurn, and the open shader toolchain that
makes in-process `.milk` ingestion possible. This file reproduces the notices
required when the bundled/linked components are redistributed inside the OjoDrop
binary. Full upstream license texts ship with the vendored sources under
`../particle-milkdrop-converter-sys/vendor/milkdrop-shader-converter/` (git
submodules) and are reproduced there verbatim.

A standalone licensing audit verified every component below is permissive and
MIT-redistribution-compatible: **no GPL/LGPL/SGI-B copyleft contamination.**

## Credits — thank you

OjoDrop would not exist without:

- **Ryan Geiss** — creator of **MilkDrop** (and the Geiss plug-in), the
  audio-reactive visualizer this engine reimplements. The per-frame/per-vertex
  equation model, warp/composite shader pipeline, and the whole aesthetic are his.
- **Jordan "jberg" Berg** — author of **Butterchurn** (the WebGL MilkDrop port)
  and **milkdrop-shader-converter**, the HLSL→GLSL toolchain OjoDrop bundles to
  read raw `.milk` presets in-process.
- **Nullsoft / Winamp** — the home MilkDrop grew up in, and the preset ecosystem
  the community built there.
- The **hlsl2glslfork**, **glsl-optimizer**, **Mesa**, and **MojoShader** authors
  (Unity Technologies, 3Dlabs, ATI, Brian Paul, Ryan C. Gordon, and contributors),
  whose shader-compiler code does the heavy lifting of converting MilkDrop's HLSL.
- Every preset author whose `.milk` files made this worth building.

## Bundled / linked components

| Component | Author(s) | License | Obligation |
|---|---|---|---|
| [Butterchurn](https://github.com/jberg/butterchurn) (preset model, reference) | Jordan Berg | MIT | notice |
| [milkdrop-shader-converter](https://github.com/jberg/milkdrop-shader-converter) | Jordan Berg | MIT | notice |
| [hlsl2glslfork](https://github.com/aras-p/hlsl2glslfork) | Unity / 3Dlabs / ATI / BitSquid | BSD-3-Clause | **reproduce notice in binaries** |
| └ MojoShader preprocessor (embedded) | Ryan C. Gordon | zlib/libpng | **reproduce notice in binaries** |
| [glsl-optimizer](https://github.com/aras-p/glsl-optimizer) | Unity Technologies | MIT | notice |
| Mesa / glcpp (within glsl-optimizer) | Brian Paul & contributors | MIT | notice |
| Bison-generated parser skeletons | FSF | GPL **+ FSF special exception** | exception removes GPL from generated output |

The `.milk` → GLSL converter is built from the vendored submodule sources at build
time (see the project README); it is not a pre-built blob. The full BSD-3, zlib,
and MIT texts live in the submodule's `LICENSE` / `LICENSE.txt` / `license.txt`
files, which are part of the redistributed source.

---

## hlsl2glslfork — BSD-3-Clause (notice reproduction required)

> Copyright (C) 2010-2014 Unity Technologies Inc.
> Copyright (C) 2005-2006 ATI Research, Inc.
> Copyright (C) 2002-2005 3Dlabs Inc. Ltd.
> Copyright (C) 2012 BitSquid AB
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
> redistributions of source code retain the above copyright notice; redistributions
> in binary form reproduce the above copyright notice in the documentation and/or
> other materials provided with the distribution; and neither the names of the
> copyright holders nor the names of contributors may be used to endorse or promote
> products derived from this software without specific prior written permission.
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
> IMPLIED WARRANTIES ARE DISCLAIMED. (Full text:
> `…/vendor/milkdrop-shader-converter/hlsl2glslfork/LICENSE.txt`.)

## MojoShader — zlib/libpng (notice reproduction required)

> Copyright (C) 2008-2016 Ryan C. Gordon and contributors.
>
> This software is provided 'as-is', without any express or implied warranty. In
> no event will the authors be held liable for any damages arising from the use of
> this software. Permission is granted to anyone to use this software for any
> purpose, including commercial applications, and to alter it and redistribute it
> freely, subject to the following restrictions: (1) the origin of this software
> must not be misrepresented; (2) altered source versions must be plainly marked as
> such; (3) this notice may not be removed or altered from any source distribution.

## glsl-optimizer / Mesa / glcpp — MIT

> Copyright (C) Unity Technologies, Brian Paul, and the Mesa contributors.
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in the
> Software without restriction… (Full text:
> `…/vendor/milkdrop-shader-converter/glsl-optimizer/license.txt`.)

## milkdrop-shader-converter — MIT

> Copyright (c) 2018 Jordan Berg. Permission is hereby granted, free of charge…
> (Full text: `…/vendor/milkdrop-shader-converter/LICENSE`.)
