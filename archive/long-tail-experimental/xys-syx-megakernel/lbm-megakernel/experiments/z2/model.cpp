// Independent scalar ownership model. No CUDA or production coordinate helpers.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "../../include/layout_config.h"
static const int dx[19]={0,0,0,1,-1,0,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1};
static const int dy[19]={0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0};
static const int dz[19]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
static bool physical(int x,int y,int z) {
    const int flat=x+PADDED_X*y+PADDED_X*PADDED_Y*z;
    return flat>=0 && flat<PADDED_X*PADDED_Y*SIZE_Z && flat%PADDED_X<SIZE_X;
}
int main() {
    std::array<uint64_t,12> counters{};
    uint64_t slots=0,cores=0,old_visits=0,old_valid=0;
    for(int bz=0;bz<38;bz+=2) for(int by=0;by<15;++by) for(int bx=0;bx<8;++bx) {
        const int ox=bx*16,oy=by*8,oz=bz*4;
        std::array<unsigned char,1800> producer_owners{};
        std::vector<unsigned char> writers(19*16*8*8,0);
        // Scalar Cartesian iteration, independent of the GPU's linear decoder.
        for(int pz=-1;pz<=8;++pz) for(int py=-1;py<=8;++py) for(int px=-1;px<=16;++px) {
            const int rank=pz<=3?0:1;
            const bool valid=physical(ox+px,oy+py,oz+pz);
            ++producer_owners[(px+1)+18*((py+1)+10*(pz+1))];
            ++counters[rank]; if(valid) ++counters[2+rank];
            for(int q=0;q<19;++q) {
                const int rx=px+dx[q],ry=py+dy[q],rz=pz+dz[q];
                if(rx<0||rx>=16||ry<0||ry>=8||rz<0||rz>=8 ||
                   ox+rx>=SIZE_X||oy+ry>=SIZE_Y||oz+rz>=SIZE_Z) continue;
                const int target=rz<4?0:1;
                const bool remote=target!=rank;
                ++writers[q*1024+rx+16*(ry+8*rz)];
                ++counters[valid?(remote?5:4):(remote?7:6)];
                if(remote) {
                    ++counters[8+rank];
                    if(bx==1&&by==1&&bz==2) ++counters[10+rank];
                    assert((rank==0&&pz==3&&dz[q]==1)||(rank==1&&pz==4&&dz[q]==-1));
                }
            }
        }
        for(auto n:producer_owners) assert(n==1);
        // Invert the scatter independently: each physical consumer has one
        // logical predecessor, one producer rank, and exactly one forward write.
        for(int rz=0;rz<8;++rz) for(int ry=0;ry<8;++ry) for(int rx=0;rx<16;++rx) {
            const bool live=ox+rx<SIZE_X&&oy+ry<SIZE_Y&&oz+rz<SIZE_Z;
            cores+=live;
            for(int q=0;q<19;++q) {
                const int px=rx-dx[q],py=ry-dy[q],pz=rz-dz[q];
                assert(px>=-1&&px<=16&&py>=-1&&py<=8&&pz>=-1&&pz<=8);
                assert(producer_owners[px+1+18*(py+1+10*(pz+1))]==1);
                assert(writers[q*1024+rx+16*(ry+8*rz)]==(live?1:0));
                slots+=live;
            }
        }
        for(int rank=0;rank<2;++rank) for(int z=-1;z<=4;++z)
            for(int y=-1;y<=8;++y) for(int x=-1;x<=16;++x) {
                ++old_visits; old_valid+=physical(ox+x,oy+y,oz+4*rank+z);
            }
    }
    assert(cores==2160000 && slots==41040000);
    assert(counters[0]==2052000 && counters[1]==2052000);
    assert(counters[10]==640 && counters[11]==640);
    printf("{\"passed\":true,\"cores\":%llu,\"slots\":%llu,\"old_candidates\":%llu,\"old_valid\":%llu,\"counts\":[",
           (unsigned long long)cores,(unsigned long long)slots,(unsigned long long)old_visits,(unsigned long long)old_valid);
    for(int i=0;i<12;++i) printf("%s%llu",i?",":"",(unsigned long long)counters[i]);
    puts("]}");
}
