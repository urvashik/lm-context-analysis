"
Licensed by Merity et al.

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"

## Modified from original code! Added support to generate POS data.

echo "=== Acquiring datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
mkdir -p penn
cd penn
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz
mv simple-examples/data/ptb.train.txt train.txt
mv simple-examples/data/ptb.test.txt test.txt
mv simple-examples/data/ptb.valid.txt valid.txt
rm -rf simple-examples.tgz
rm -rf simple-examples/
cd ..

echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt

cd ..
cd ..

echo "- Generating POS tagged data"
mkdir -p data/penn_pos
python get_pos_data.py penn
mkdir -p data/wikitext-2_pos
python get_pos_data.py wikitext-2

echo "---"
echo "Happy language modeling :)"
