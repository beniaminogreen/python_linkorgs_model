{ pkgs ? import <nixpkgs> {} }: with pkgs; mkShell { buildInputs = [
    python311Packages.torchvision
    python311Packages.torchaudio
    python311Packages.tqdm
    python311Packages.ray
    stdenv.cc.cc.lib
];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
  '';
}
