#!/usr/bin/env python
import argparse
import subprocess
import pipes
import sys
import os
import re

def get_file(path):
    with open(path, "r") as f:
        return str(f.read())


def quote(v, ):
    if isinstance(v, list):
        return "[" + ", ".join(quote(x) for x in v) + "]"
    else:
        v = pipes.quote(v)
        if not v.startswith("'") or not v.startswith('"'):
            v = '"' + v + '"'
        return v


def variable_sub(s, table, no_quote):

    for k in table:
        if k not  in no_quote:
            t = quote(table[k])
        else:
            t = table[k]
        s = s.replace("${"+k+"}", t)
    return s


def parse_mem(mem):
    m = re.search("(\d+[KMGT]?)(I)?(B)?", mem.upper())
    if m is None:
        raise "{} is invalid size".format(mem)

    return m.group(1)

def get_mem_str(mem):
    if mem is None:
        return ""
    return "memory: {}i".format(parse_mem(mem))


def generate_yaml(type, cmd, name, mem_limit, mem_req, node, ngpus, ssh_key, user):
    variables={}
    variables["CMD"] = cmd
    variables["NAME"] = name
    ssh_cmd = "eval \\\"$(ssh-agent -s)\\\";  ssh-add /data/ssh-keys/ssh-{}-rsa; chmod 600 /data/ssh-keys/ssh-{}-rsa.pub; chmod 600 /data/ssh-keys/ssh-${}-rsa; mkdir /root/.ssh; ssh-keyscan github.com >> /root/.ssh/known_hosts".format(user,user,user)
    variables["PIP_CMD"] = "pip3 install opencv-contrib-python; pip3 install easydict; pip3 install pyyaml; pip3 install hdf5storage;  pip install metayaml; pip install deepdish; pip install -U scikit-learn; pip3 install jupyter; pip3 install --upgrade torch; pip3 install tensorboard; pip3 install opencv-python; pip3 install metayaml; pip3 install tb-nightly;pip3 install torchnet; pip3 install sklearn; pip3 install seaborn"
    variables["UNZIP_CMD"] = "echo Unzipping; mkdir -p /localdata/domainnet; unzip /data-domainnet/clipart.zip  -d /localdata/domainnet/;unzip /data-domainnet/infograph.zip  -d /localdata/domainnet/;unzip /data-domainnet/painting.zip  -d /localdata/domainnet/;unzip /data-domainnet/quickdraw.zip  -d /localdata/domainnet/;unzip /data-domainnet/real.zip  -d /localdata/domainnet/;unzip /data-domainnet/sketch.zip  -d /localdata/domainnet/;cp -r /data-domainnet/txt /localdata/domainnet/; "
    variables["NGPUS"] = str(ngpus)
    variables["SSH_CMD"] = ssh_cmd


    variables["MEM_LIMIT"] = get_mem_str(mem_limit)
    variables["MEM_REQ"] = get_mem_str(mem_req)
    variables["NODE_HOSTNAME"] = "kubernetes.io/hostname: " + quote(node) if node is not None else ""

    variables["SSH_KEY_NAME"] = "secretName: " + quote("ssh-key-secret-" + ssh_key) if ssh_key else ""

    script_dir = os.path.dirname(os.path.realpath(__file__))
    return (variable_sub(get_file(os.path.join(script_dir, type + ".yaml")), variables, {"MEM_LIMIT", "MEM_REQ", "NODE_HOSTNAME", "SSH_KEY_NAME", "NGPUS","SSH_CMD", "PIP_CMD", "UNZIP_CMD", "CMD"}))


def run_cmd(cmd, stdin_str):
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr,stdin=subprocess.PIPE)

    out, err = proc.communicate(input=stdin_str)
    result = proc.returncode == 0

    if result == False:
        print("Error")

    return proc.returncode


def gen_kubectl_run_cmd(podname, type, repo=None, branch=None, cmds=None, mem_limit=None, mem_req=None, node=None, ngpus=1, key=None, user=None):
    kube_cmd = ["kubectl", "create", "-f", "-"]

    if False:
        kube_cmd.append("--dry-run")

    yaml_str = (generate_yaml(type, cmds, podname, mem_limit, mem_req, node, ngpus, key, user))
    print(yaml_str)
    return run_cmd(kube_cmd, yaml_str)


def get_arg(args):
    if args is not None:
        return args[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Deploying command')

    parser.add_argument('--name', metavar='pod_name', type=str, nargs=1,
                        help='Pod name', required=True)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--repo', metavar='my_git_repo', type=str, nargs=1,
                        help='Path to git repo')

    parser.add_argument('--branch', metavar='git_branch', type=str, nargs=1,
                        help='Git branch', default=["master"])

    group.add_argument('--empty',  action='store_true',
                        help='Creates pods without repo')

    parser.add_argument('--type',  type=str, nargs=1,
                        help='Type of pod', default=['deeplearning'])
    parser.add_argument('--user',  type=str, nargs=1, choices=['nitesh', 'shreyas'],
                        help='User for ssh')

    parser.add_argument('--ngpus',  type=int,
                        help='Number of gpus', default=1)

    parser.add_argument("--command", type=str, help='Command', default='')

    parser.add_argument('--meml', type=str, nargs=1,
                        help='Memory limit')

    parser.add_argument('--memr', type=str, nargs=1,
                        help='Memory request')

    parser.add_argument('--node', type=str, nargs=1,
                        help='Node name')

    parser.add_argument('--key', type=str, nargs=1,
                        help='SSH key name for git', default=[None])

    args = parser.parse_args()

    if args.branch is None and args.repo is None and len(args.command) == 0:
        args.empty = True

    ret_code = gen_kubectl_run_cmd(args.name[0], args.type[0], get_arg(args.repo), get_arg(args.branch), args.command, mem_limit=get_arg(args.meml), mem_req=get_arg(args.memr), node=get_arg(args.node), ngpus=args.ngpus, key=args.key[0], user=get_arg(args.user))
    sys.exit(ret_code)

if __name__ == main():
    main()
