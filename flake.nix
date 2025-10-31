{
  description = "Giggle ML";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-old.url = "github:NixOS/nixpkgs/nixos-21.05";
    flake-utils.url = "github:numtide/flake-utils";
    seqpare-src = {
      url = "github:deepstanding/seqpare";
      flake = false;
    };
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-old,
      flake-utils,
      seqpare-src,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-old = nixpkgs-old.legacyPackages.${system};

        seqpare = pkgs.stdenv.mkDerivation {
          pname = "seqpare";
          version = "unstable";
          src = seqpare-src;

          buildInputs = with pkgs; [
            gnumake
            gcc
            zlib
          ];

          buildPhase = ''
            make --version
            make
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp bin/seqpare $out/bin/
          '';
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            seqpare
            uv
            just
            bedtools
            # wget
            # bedtools
            # samtools
          ];
        };
      }
    );
}
