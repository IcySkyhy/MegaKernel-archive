// Prospective producer counts only: C0 itself still computes independent halos.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include "../../include/layout_config.h"
static bool valid(int x,int y,int z) {
    int s=x+PADDED_X*(y+PADDED_Y*z);
    return s>=0&&s<PADDED_X*PADDED_Y*SIZE_Z&&s%PADDED_X<SIZE_X;
}
int main() {
    const int shapes[][3]={{1,1,2},{2,1,1},{1,3,1},{2,1,2},{1,3,2},{4,1,2}};
    const char *names[]={"Z2","X2","Y3","XZ4","YZ6","X4Z2"};
    uint64_t old_candidates=0,old_valid=0;
    for(int z=0;z<38;++z)for(int y=0;y<15;++y)for(int x=0;x<8;++x)
        for(int pz=-1;pz<=4;++pz)for(int py=-1;py<=8;++py)for(int px=-1;px<=16;++px) {
            ++old_candidates;old_valid+=valid(x*16+px,y*8+py,z*4+pz);
        }
    printf("{\"old_candidates\":%llu,\"old_valid\":%llu,\"shapes\":{",(unsigned long long)old_candidates,(unsigned long long)old_valid);
    for(int k=0;k<6;++k) {
        auto &c=shapes[k];uint64_t candidates=0,physical=0;
        assert(8%c[0]==0&&15%c[1]==0&&38%c[2]==0);
        for(int z=0;z<38;z+=c[2])for(int y=0;y<15;y+=c[1])for(int x=0;x<8;x+=c[0])
            for(int pz=-1;pz<=4*c[2];++pz)for(int py=-1;py<=8*c[1];++py)for(int px=-1;px<=16*c[0];++px) {
                ++candidates;physical+=valid(x*16+px,y*8+py,z*4+pz);
            }
        printf("%s\"%s\":{\"candidates\":%llu,\"valid\":%llu}",k?",":"",names[k],(unsigned long long)candidates,(unsigned long long)physical);
    }
    puts("}}");
}
