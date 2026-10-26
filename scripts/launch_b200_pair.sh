#!/bin/bash
# Launch 2x p6-b200.48xlarge spot in ap-south-2b, cluster placement group, 8 EFA interfaces each, Elastic IPs.
#   bash launch_b200_pair.sh [--dry-run] [--loop SECONDS] [--tries N]
set -uo pipefail
R=ap-south-2; AZ=ap-south-2b; SUBNET=subnet-089990216eeb3c495; VPC=vpc-0c55595d32c61836e
AMI=ami-0652dbb11d43ca6f7   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20261017
KEY=henanwan-ap-south-2; TYPE=p6-b200.48xlarge; PG=b200-bench-pg; SGNAME=b200-bench; NAME=b200-bench
LAPTOP_IP=54.240.199.100/32; JUMP_IP=3.101.143.60/32
DRY=""; LOOP=60; TRIES=60
while [ $# -gt 0 ]; do case $1 in --dry-run) DRY=--dry-run;; --loop) LOOP=$2; shift;; --tries) TRIES=$2; shift;; esac; shift; done

SG=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=$SGNAME Name=vpc-id,Values=$VPC --query "SecurityGroups[0].GroupId" --output text)
if [ "$SG" = None ] || [ -z "$SG" ]; then
    SG=$(aws ec2 create-security-group --region $R --group-name $SGNAME --description "B200 bench: ssh from laptop/Jump, all traffic within the group (EFA)" --vpc-id $VPC --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --region $R --group-id $SG --ip-permissions \
        "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$LAPTOP_IP},{CidrIp=$JUMP_IP}]" \
        "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG}]" > /dev/null
    aws ec2 authorize-security-group-egress --region $R --group-id $SG --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG}]" > /dev/null
fi
echo "SG $SG"
aws ec2 describe-placement-groups --region $R --group-names $PG > /dev/null 2>&1 || aws ec2 create-placement-group --region $R --group-name $PG --strategy cluster > /dev/null
echo "PG $PG"

NI=""
for i in 0 1 2 3 4 5 6 7; do
    dev=$([ $i = 0 ] && echo 0 || echo 1)
    NI="$NI NetworkCardIndex=$i,DeviceIndex=$dev,InterfaceType=efa,SubnetId=$SUBNET,Groups=$SG,DeleteOnTermination=true"
done
for t in $(seq $TRIES); do
    out=$(aws ec2 run-instances --region $R $DRY --image-id $AMI --instance-type $TYPE --key-name $KEY --count 2 \
        --placement "AvailabilityZone=$AZ,GroupName=$PG" \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --network-interfaces $NI \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --query "Instances[].InstanceId" --output text 2>&1)
    rc=$?
    echo "$(date +%H:%M:%S) try $t rc=$rc: $out" | cut -c1-300
    [ -n "$DRY" ] && exit 0
    [ $rc = 0 ] && break
    echo "$out" | grep -qE "InsufficientInstanceCapacity|MaxSpotInstanceCountExceeded|SpotMaxPriceTooLow|capacity" || exit 1
    sleep $LOOP
done
[ $rc = 0 ] || exit 1
IDS=$out
aws ec2 wait instance-running --region $R --instance-ids $IDS
for id in $IDS; do
    eni=$(aws ec2 describe-instances --region $R --instance-ids $id --query "Reservations[0].Instances[0].NetworkInterfaces[?Attachment.NetworkCardIndex==\`0\` && Attachment.DeviceIndex==\`0\`].NetworkInterfaceId|[0]" --output text)
    alloc=$(aws ec2 allocate-address --region $R --domain vpc --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" --query AllocationId --output text)
    aws ec2 associate-address --region $R --allocation-id $alloc --network-interface-id $eni > /dev/null
    ip=$(aws ec2 describe-addresses --region $R --allocation-ids $alloc --query "Addresses[0].PublicIp" --output text)
    priv=$(aws ec2 describe-instances --region $R --instance-ids $id --query "Reservations[0].Instances[0].PrivateIpAddress" --output text)
    echo "INSTANCE $id public $ip private $priv eip $alloc"
done
