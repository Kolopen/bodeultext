export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

# macOS 기본 sed/awk/sort 는 Kaldi 스크립트가 기대하는 GNU 판과 동작이 다르다.
# brew install coreutils gnu-sed gawk grep 으로 깔고 앞에 둔다.
if [ "$(uname)" = Darwin ] && command -v brew >/dev/null 2>&1; then
  for pkg in coreutils gnu-sed grep findutils; do
    gnubin="$(brew --prefix)/opt/$pkg/libexec/gnubin"
    [ -d "$gnubin" ] && export PATH="$gnubin:$PATH"
  done
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1
