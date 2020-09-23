# poker

```bash
sudo docker build ~/poker --tag=poker_docker
sudo docker run -it -v ~/poker:/home/poker poker_docker bash
pip install -e .
python poker/play.py
```